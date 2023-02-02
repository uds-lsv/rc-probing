
import torch
from torch import nn

from transformers.modeling_bert import BertLayer, BertEncoder, BertAttention, BertOutput, BertSelfOutput, BertModel, BertForSequenceClassification

from vectorizer.models.adapters import FeedForwardAdapter


class BertSelfOutputWithAdapter(BertSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        # Adapter module with bottleneck layer
        self.adapter = FeedForwardAdapter(
            config.hidden_size, config.adapter_dim, config.adapter_init_std)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # add adapter to forward pass
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertOutputWithAdapter(BertOutput):
    def __init__(self, config):
        super().__init__(config)
        # Adapter module with bottleneck layer
        self.adapter = FeedForwardAdapter(
            config.hidden_size, config.adapter_dim, config.adapter_init_std)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # add adapter to forward pass
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttentionWithAdapter(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.output = BertSelfOutputWithAdapter(config)


class BertLayerWithAdapters(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionWithAdapter(config)
        if self.is_decoder:
            self.crossattention = BertAttentionWithAdapter(config)
        self.output = BertOutputWithAdapter(config)


class BertEncoderWithAdapters(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertLayerWithAdapters(
            config) for _ in range(config.num_hidden_layers)])


class BertModelWithAdapters(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoderWithAdapters(config)


class BertForSequenceClassificationWithAdapters(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelWithAdapters(config)
