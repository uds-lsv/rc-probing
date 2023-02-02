import math
import torch
from torch import nn

from transformers.modeling_albert import AlbertAttention, AlbertLayer, AlbertLayerGroup, AlbertTransformer, AlbertModel, AlbertForSequenceClassification

from vectorizer.models.adapters import FeedForwardAdapter


class AlbertAttentionWithAdapter(AlbertAttention):
    def __init__(self, config):
        super().__init__(config)

        # Adapter module with bottleneck layer
        self.adapter = FeedForwardAdapter(
            config.hidden_size, config.adapter_dim, config.adapter_init_std)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(input_ids)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum(
            "bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)

        # add adapter to forward pass
        projected_context_layer_dropout = self.adapter(
            projected_context_layer_dropout)

        layernormed_context_layer = self.LayerNorm(
            input_ids + projected_context_layer_dropout)
        output = (layernormed_context_layer, attention_probs) if self.output_attentions else (
            layernormed_context_layer,)

        return output


class AlbertLayerWithAdapters(AlbertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = AlbertAttentionWithAdapter(config)

        # Adapter module with bottleneck layer
        self.adapter = FeedForwardAdapter(
            config.hidden_size, config.adapter_dim, config.adapter_init_std)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_output = self.attention(
            hidden_states, attention_mask, head_mask)
        ffn_output = self.ffn(attention_output[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        ffn_output = self.adapter(ffn_output)  # add adapter to forward pass
        hidden_states = self.full_layer_layer_norm(
            ffn_output + attention_output[0])

        # add attentions if we output them
        return (hidden_states,) + attention_output[1:]


class AlbertLayerGroupWithAdapters(AlbertLayerGroup):
    def __init__(self, config):
        super().__init__(config)
        self.albert_layers = nn.ModuleList(
            [AlbertLayerWithAdapters(config) for _ in range(config.inner_group_num)])


class AlbertTransformerWithAdapters(AlbertTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.albert_layer_groups = nn.ModuleList(
            [AlbertLayerGroupWithAdapters(config) for _ in range(config.num_hidden_groups)])


class AlbertModelWithAdapters(AlbertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = AlbertTransformerWithAdapters(config)


class AlbertForSequenceClassificationWithAdapters(AlbertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.albert = AlbertModelWithAdapters(config)
