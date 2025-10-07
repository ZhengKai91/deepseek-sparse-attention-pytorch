from typing import Callable, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.deprecation import deprecate_kwarg

# 本项目自定义配置
from .configuration_qwen3_dsa import Qwen3DSAConfig


def fp16_index(q, weights, k):
    # q: (bsz, seqlen, n_heads, head_dim)
    # weights: (bsz, seq_len, n_heads, 1)
    # k: (bsz, seqlen_k, head_dim)
    index_score = torch.einsum(
        "bsnd,btd->bsnt", q, k
    )  # (bsz, seqlen, n_heads, seqlen_k)
    index_score = F.relu(index_score)
    weighted = index_score * weights  # (bsz, seqlen, n_heads, seqlen_k)
    index_score = weighted.sum(dim=2)  # (bsz, seqlen, seqlen_k)
    return index_score


class Indexer(nn.Module):
    def __init__(self, config: Qwen3DSAConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.head_dim, bias=False)
        self.w_proj = nn.Linear(self.d_model, self.n_heads, bias=False)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.softmax_scale = self.head_dim**-0.5
        self.k_cache = None
        self.max_seq_len = config.max_position_embeddings

    def update(self, k, start_pos, end_pos):
        bsz, seqlen, _ = k.shape
        assert seqlen == end_pos - start_pos, "k length must match [start_pos, end_pos)"
        if self.k_cache is None:
            self.k_cache = torch.zeros(
                bsz, self.max_seq_len, self.head_dim, dtype=k.dtype, device=k.device
            )
        self.k_cache[:, start_pos:end_pos] = k

    def forward(
        self, x: torch.Tensor, start_pos: int, end_pos: int, **kwargs
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        hidden_shape = (bsz, seqlen, -1, self.head_dim)
        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)

        position_embeddings = kwargs["sparse_position_embeddings"]
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # (bsz, n_heads, seqlen, head_dim)
        q = q.transpose(1, 2)  # (bsz, seqlen, n_heads, head_dim)
        k = k.transpose(1, 2).squeeze(2)  # (bsz, seqlen, head_dim)

        if kwargs["use_cache"] and start_pos >= 0 and end_pos >= 0:
            self.update(k, start_pos, end_pos)
            k = self.k_cache[:, :end_pos]

        weights = self.w_proj(x) * self.n_heads**-0.5  # (bsz,  n_heads)
        weights = (
            weights.unsqueeze(-1) * self.softmax_scale
        )  # (bsz, seqlen, n_heads, 1)

        index_score = fp16_index(q, weights, k)  # (bsz, seqlen, seqlen)

        # when use cache, mask may not seqlen, seqlen, to fix
        seqlen_k = index_score.shape[-1]
        mask = (
            torch.full((seqlen, seqlen_k), float("-inf"), device=x.device).triu_(1)
            if seqlen > 1
            else None
        )
        if mask is not None:
            index_score += mask

        topk_indices = index_score.topk(
            min(self.index_topk, index_score.shape[-1]), dim=-1
        )[1]
        # topk_indices_ = topk_indices.clone()
        # dist.broadcast(topk_indices_, src=0)
        # assert torch.all(topk_indices == topk_indices_), f"{topk_indices=} {topk_indices_=}"
        return topk_indices


class Qwen3DSAAttention(Qwen3Attention):
    def __init__(self, config: Qwen3DSAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.indexer = Indexer(config)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )  # bsz, n_heads, seqlen, head_dim

        # if self.layer_idx ==1:
        #     import pdb; pdb.set_trace()

        seqlen = key_states.shape[-2]
        start_pos, end_pos = 0, seqlen
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            start_pos, end_pos = key_states.shape[-2] - seqlen, key_states.shape[-2]

        topk_indices = self.indexer(hidden_states, start_pos, end_pos, **kwargs)
        mask_shape = (*input_shape, key_states.shape[-2])
        mask = (
            torch.full(mask_shape, float("-inf"), device=hidden_states.device).triu_(1)
            if mask_shape[-2] > 1
            else None
        )
        index_mask = torch.full(
            mask_shape, float("-inf"), device=hidden_states.device
        ).scatter_(-1, topk_indices, 0)
        if mask is not None:
            index_mask += mask
        index_mask = index_mask.unsqueeze(1)
        if attention_mask is not None:
            index_mask = attention_mask + index_mask

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        # is_causal if infer by:  query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            index_mask,  # attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DSAModel(Qwen3Model):
    def __init__(self, config: Qwen3DSAConfig):
        super().__init__(config)
        ori_head_dim = config.head_dim
        config.head_dim = config.index_head_dim
        self.sparse_rotary_emb = Qwen3RotaryEmbedding(config)
        config.head_dim = ori_head_dim

        for layer in self.layers:
            old_attn = layer.self_attn
            new_attn = Qwen3DSAAttention(config, layer_idx=old_attn.layer_idx)
            layer.self_attn = new_attn

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping[
                    "sliding_attention"
                ] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        sparse_position_embeddings = self.sparse_rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                sparse_position_embeddings=sparse_position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen3DSAForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    config_class = Qwen3DSAConfig
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Qwen3DSAConfig):
        super().__init__(config)
        self.model = Qwen3DSAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        if not hasattr(config, "sparse_lambda"):
            config.sparse_lambda = 0.01

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        # # 聚合 sparse loss
        # if kwargs.get("labels") is not None:
        #     sparse_loss = sum(
        #         layer.self_attn.get_sparse_loss()
        #         for layer in self.model.layers
        #     )
        #     outputs.loss = outputs.loss + self.config.sparse_lambda * sparse_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_qwen3_dsa_from_pretrained(model_name_or_path, config=None, **kwargs):
    if config is None:
        base_config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        config = Qwen3DSAConfig.from_dict(base_config.to_dict())

    base_model = Qwen3ForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    custom_model = Qwen3DSAForCausalLM(config)
    custom_model.load_state_dict(base_model.state_dict(), strict=False)

    assert torch.allclose(
        custom_model.model.embed_tokens.weight,
        base_model.model.embed_tokens.weight,
        atol=1e-6,
    )
    for i in range(min(2, len(base_model.model.layers))):
        assert torch.allclose(
            base_model.model.layers[i].self_attn.q_proj.weight,
            custom_model.model.layers[i].self_attn.q_proj.weight,
            atol=1e-6,
        )

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return custom_model
