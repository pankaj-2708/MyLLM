from transformers import PretrainedConfig


class PM_MiniFinLLM_config(PretrainedConfig):
    model_type = "PM_MiniFinLLM"

    def __init__(
        self, vocab_size=32000, embed_dim=768, n_layer=12, n_head=12, **kwargs
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_layer = n_layer
        self.n_head = n_head
        super().__init__(**kwargs)
