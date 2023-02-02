import h5py

from tqdm import tqdm
import numpy as np

from transformers import (WEIGHTS_NAME,
                          AutoConfig, AutoTokenizer, AutoModel, AutoModelWithLMHead,
                          BertConfig, BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, BertForQuestionAnswering,
                          DistilBertConfig, DistilBertTokenizer, DistilBertModel,
                          RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMaskedLM, RobertaForSequenceClassification, RobertaForQuestionAnswering,
                          AlbertConfig, AlbertTokenizer, AlbertModel, AlbertForMaskedLM, AlbertForSequenceClassification, AlbertForQuestionAnswering,
                          GPT2Config, GPT2Tokenizer, GPT2Model,
                          XLNetConfig, XLNetTokenizer, XLNetModel,
                          T5Config, T5Tokenizer, T5Model,
                          )

from vectorizer.models.adapter_bert import BertForSequenceClassificationWithAdapters
from vectorizer.models.adapter_roberta import RobertaForSequenceClassificationWithAdapters
from vectorizer.models.adapter_albert import AlbertForSequenceClassificationWithAdapters

HF_HOSTED_MODELS = {
    'google/electra-base',
}

ENCODER_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'bert-mlm': (BertConfig, BertForMaskedLM, BertTokenizer),
    'bert-finetuned': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'bert-finetuned-squad': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'adapter-bert-finetuned': (BertConfig, BertForSequenceClassificationWithAdapters, BertTokenizer),
    'distilbert': (DistilBertConfig, DistilBertModel, DistilBertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'roberta-mlm': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'roberta-finetuned': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'roberta-finetuned-squad': (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
    'adapter-roberta-finetuned': (RobertaConfig, RobertaForSequenceClassificationWithAdapters, RobertaTokenizer),
    'meanpooling-roberta-finetuned': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'fivepooling-roberta-finetuned': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilroberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertModel, AlbertTokenizer),
    'albert-mlm': (AlbertConfig, AlbertForMaskedLM, AlbertTokenizer),
    'albert-finetuned': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    'albert-finetuned-squad': (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
    'adapter-albert-finetuned': (AlbertConfig, AlbertForSequenceClassificationWithAdapters, AlbertTokenizer),
}


def load_pretrained_model_and_tokenizer(model_type, model_name_or_path, config_name=None, tokenizer_name=None, do_lower_case=False,
                                        output_hidden_states=True, output_attentions=True, cache_dir=None):
    # assert model_type in ENCODER_CLASSES

    if model_name_or_path in HF_HOSTED_MODELS:
        raise NotImplementedError()
        # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # if 'electra' in model_name_or_path:
        #     model = AutoModelWithLMHead.from_pretrained(model_name_or_path)
        # else:
        #     model = AutoModel.from_pretrained(model_name_or_path)

        # model_config = model.config

    else:
        # Get config, model_class and tokenizer for model_type
        config_class, model_class, tokenizer_class = ENCODER_CLASSES[model_type]

        model_config = config_class.from_pretrained(
            pretrained_model_name_or_path=config_name if config_name else model_name_or_path,
            cache_dir=cache_dir if cache_dir else None,
        )

        model_config.output_hidden_states = output_hidden_states
        model_config.output_attentions = output_attentions

        # Load pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path,
            do_lower_case=do_lower_case,
            cache_dir=cache_dir if cache_dir else None,
        )

        # Models can return full list of hidden-states & attentions weights at each layer
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,
            cache_dir=cache_dir if cache_dir else None,
        )

        # For fine-tuned models return only the encoder
        if model_type in ['bert-finetuned', 'bert-finetuned-squad', 'adapter-bert-finetuned']:
            model = model.bert

        if model_type in ['albert-finetuned', 'albert-finetuned-squad', 'adapter-albert-finetuned']:
            model = model.albert

        if model_type in ['roberta-finetuned', 'roberta-finetuned-squad', 'adapter-roberta-finetuned', 'meanpooling-roberta-finetuned', 'fivepooling-roberta-finetuned']:
            model = model.roberta

    return model, model_config, tokenizer


def load_embeddings_from_hdf5(input_file):
    # the hdf5 embedding files are dictionaries of length n_samples containing numpy arrays of shape (embedding_dim, )
    f = h5py.File(input_file, 'r')

    # NOTE: hdf5 file keys are sorted *alphabetically*: ['0', '1', '10', '12', ... ]. This will cause issues when assigning labels to samples.
    rows = list(f.keys())
    rows.sort(key=int)  # sort them numerically

    embeddings_list = []
    for row in tqdm(rows, desc='Reading embeddings'):
        embedding = np.asarray(f[row])
        embeddings_list.append(embedding)

    embeddings = np.asarray(embeddings_list).squeeze()

    return embeddings


def load_labels_from_npy(input_file):
    labels = np.load(input_file, allow_pickle=True)
    return labels
