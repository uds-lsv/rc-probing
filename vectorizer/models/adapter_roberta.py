import torch
from torch import nn

from transformers.modeling_roberta import RobertaEmbeddings, RobertaModel, RobertaForSequenceClassification

from vectorizer.models.adapter_bert import BertModelWithAdapters


class RobertaModelWithAdapters(BertModelWithAdapters):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


class RobertaForSequenceClassificationWithAdapters(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModelWithAdapters(config)
