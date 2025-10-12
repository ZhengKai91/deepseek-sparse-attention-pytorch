# modeling_loader.py
import os
import torch
from transformers import AutoConfig
from models.modeling_qwen3_dsa import Qwen3DSAConfig, Qwen3DSAForCausalLM
from transformers import Qwen3ForCausalLM


def load_qwen3_dsa_from_pretrained(model_name_or_path, config=None, is_ds_zero3=False, **kwargs):
    if config is None:
        base_config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        config = Qwen3DSAConfig.from_dict(base_config.to_dict())

    base_model = Qwen3ForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    custom_model = Qwen3DSAForCausalLM(config)
    custom_model = custom_model.to(base_model.dtype)

    if is_ds_zero3:
        from deepspeed.runtime.zero.partition_parameters import GatheredParameters
        with GatheredParameters(base_model.parameters(), modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                custom_model.load_state_dict(base_model.state_dict(), strict=False)
    else:
        custom_model.load_state_dict(base_model.state_dict(), strict=False)

    def assert_close(p1, p2, atol=1e-6):
        if is_ds_zero3:
            with GatheredParameters([p1, p2], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    assert torch.allclose(p1.cpu().float(), p2.cpu().float(), atol=atol)
        else:
            assert torch.allclose(p1.cpu().float(), p2.cpu().float(), atol=atol)

    assert_close(custom_model.model.embed_tokens.weight,
                 base_model.model.embed_tokens.weight)

    for i in range(min(2, len(base_model.model.layers))):
        assert_close(
            base_model.model.layers[i].self_attn.q_proj.weight,
            custom_model.model.layers[i].self_attn.q_proj.weight,
        )

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return custom_model