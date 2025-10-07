from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class Qwen3DSAConfig(Qwen3Config):
    model_type = "qwen3_dsa"

    def __init__(
        self,
        use_sparse_indexer: bool = True,
        sparse_lambda: float = 0.01,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.use_sparse_indexer = use_sparse_indexer
        self.sparse_lambda = sparse_lambda
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
