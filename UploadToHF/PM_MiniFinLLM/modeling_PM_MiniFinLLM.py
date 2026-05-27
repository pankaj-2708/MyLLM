from transformers import PreTrainedModel, GenerationMixin
from .configuration_PM_MiniFinLLM import PM_MiniFinLLM_config
from transformers.modeling_outputs import CausalLMOutputWithPast
from .arch import LLM, vocab_size, embed_dim, n_layer, n_head
import torch.nn as nn
import torch


class PM_MiniFinLLM_Model(PreTrainedModel, GenerationMixin):
    config_class = PM_MiniFinLLM_config
    _tied_weights_keys = {
        "model.linear.weight": "model.embedding.weight"  #  duplicate -> source
    }

    def __init__(self, config):
        super().__init__(config)
        self.model = LLM(
            config.vocab_size, config.embed_dim, config.n_layer, config.n_head
        )
        self.post_init()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def loss_fn(self, y_pred, y_true):
        return self.cross_entropy_loss(y_pred.reshape(-1, 32000), y_true.reshape(-1))

    def tie_weights(self, *args, **kwargs):
        self.model.linear.weight = self.model.embedding.weight

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def forward(self, input_ids, labels=None, **kwargs):
        logits = self.model(input_ids)
        loss = None

        if labels is not None:
            shifted_logits = logits[:, :-1, :]
            shifted_labels = labels[:, 1:]
            loss = self.loss_fn(shifted_logits, shifted_labels)

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)
