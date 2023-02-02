import h5py

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from allennlp.commands.elmo import ElmoEmbedder

from vectorizer.utils import load_pretrained_model_and_tokenizer

from flair.data import Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings


class Vectorizer(object):
    def __init__(self):
        pass

    def vectorize_features(self, features):
        embeddings = None
        for feature in tqdm(features, desc=f'Vectorizing features'):
            feature_embeddings = self.vectorize_feature(feature)
            # feature_embeddings = np.expand_dims(feature_embeddings, axis=0)

            if embeddings is None:
                embeddings = feature_embeddings
            else:
                embeddings = np.append(embeddings, feature_embeddings, axis=0)

        return embeddings

    def vectorize_dataset(self, dataloader, token_level_label_mode, pad_token_id=0, pad_token_label_id=-100):
        embeddings = None
        for batch in tqdm(dataloader, desc=f'Vectorizing dataset'):
            batch_embeddings = self.vectorize_batch(
                batch, token_level_label_mode, pad_token_id, pad_token_label_id)

            if embeddings is None:
                embeddings = batch_embeddings
            else:
                embeddings = np.append(embeddings, batch_embeddings, axis=0)

        return embeddings

    def make_hdf5_file_from_embeddings(self, embeddings, output_file):
        sample_idx = 0
        print(f'Saving embeddings {embeddings.shape} to: {output_file}')

        with h5py.File(output_file, 'w') as f:
            for sent_embedding in tqdm(embeddings, desc='Creating hdf5 file'):
                # embeddings are indexed by their corresponding sample index. Which is a str. Be careful when loading the data. One has to use the same str index to get the sample back.
                f.create_dataset(str(sample_idx), sent_embedding.shape,
                                 dtype='float32', data=sent_embedding)
                sample_idx += 1


class FlairVectorizer(Vectorizer):

    # TODO(mm): For token level probing tasks we need to apply padding to input sentences based on max_length

    def vectorize_feature(self, feature):
        # here feature is of type InputExample
        sent = feature.text_a
        if feature.text_b is not None:
            sent += feature.text_b

        sent = Sentence(sent)
        self.model.embed(sent)  # embed sentence

        # collect token embeddings
        hidden_states = None
        for token in sent:
            hidden_state = token.embedding.detach().cpu().numpy()
            hidden_state = np.expand_dims(hidden_state, axis=0)

            if hidden_states is None:
                hidden_states = hidden_state
            else:
                hidden_states = np.append(hidden_states, hidden_state, axis=0)

        hidden_states = np.expand_dims(hidden_states, axis=0)
        # shape: (1, seq_len, hidden_dim)

        if self.pooler == 'cls':
            hidden_states = hidden_states[:, 0, :]
        elif self.pooler == 'mean':
            hidden_states = np.mean(hidden_states, axis=1)
        elif self.pooler == 'max':
            hidden_states = np.max(hidden_states, axis=1)
        elif self.pooler == 'min':
            hidden_states = np.min(hidden_states, axis=1)

        # shape: (1, hidden_dim)

        return hidden_states


class GloveVectorizer(FlairVectorizer):
    def __init__(self, pooler=None):
        self.pooler = pooler
        if self.pooler is None:
            raise NotImplementedError(
                'GloveVectorizer currently supports only sentence emebeddings.')

        # init standard GloVe embedding
        embedding = WordEmbeddings('glove')

        # create a StackedEmbedding object that combines glove and forward/backward flair embeddings
        self.model = StackedEmbeddings([
            embedding,
        ])


class FastTextVectorizer(FlairVectorizer):
    def __init__(self, pooler=None):
        self.pooler = pooler
        if self.pooler is None:
            raise NotImplementedError(
                'FastTextVectorizer currently supports only sentence emebeddings.')

        # FastText embeddings over news and wikipedia data
        embedding = WordEmbeddings('en')

        # create a StackedEmbedding object that combines glove and forward/backward flair embeddings
        self.model = StackedEmbeddings([
            embedding,
        ])


class FlairEmbeddingsVectorizer(FlairVectorizer):
    # See: https://www.aclweb.org/anthology/C18-1139/. Especially Figure 2.
    def __init__(self, pooler=None):
        self.pooler = pooler
        if self.pooler is None:
            raise NotImplementedError(
                'FlairEmbeddingsVectorizer currently supports only sentence emebeddings.')

        # init Flair forward and backwards embeddings
        flair_embedding_forward = FlairEmbeddings('news-forward')
        flair_embedding_backward = FlairEmbeddings('news-backward')

        # create a StackedEmbedding object that combines glove and forward/backward flair embeddings
        self.model = StackedEmbeddings([
            flair_embedding_forward,
            flair_embedding_backward,
        ])


class ElmoVectorizer(Vectorizer):
    # TODO(mm): Can we speed-up ELMo? Look into AllenNLP
    # TODO(mm): How to apply padding for ELMo?
    # TODO(mm): Support combining embeddings across layers, e.g. taking mean over layer dim

    def __init__(self, layer=None, pooler=None, cuda_device=-1):
        self.model = ElmoEmbedder(cuda_device=cuda_device)
        self.layer = layer
        self.pooler = pooler
        if self.pooler is None:
            # TODO(mm): If we wan't to train a model on top of token level embeddings we need to apply padding. Same for static embeddings
            raise NotImplementedError(
                'ElmoVectorizer currently supports only sentence emebeddings.')

    def vectorize_feature(self, feature):
        # here feature is of type InputExample
        sent = feature.text_a.split()  # convert text_a to list of tokens
        if feature.text_b is not None:
            sent += feature.text_b.split()

        hidden_states = self.model.embed_sentence(sent)
        # shape: (n_layers, seq_len, hidden_dim)

        if self.layer is not None:
            hidden_states = hidden_states[self.layer]
            hidden_states = np.expand_dims(hidden_states, axis=0)
            # shape: (1, seq_len, hidden_dim)

        if self.pooler == 'cls':
            hidden_states = hidden_states[:, 0, :]
        elif self.pooler == 'mean':
            hidden_states = np.mean(hidden_states, axis=1)
        elif self.pooler == 'max':
            hidden_states = np.max(hidden_states, axis=1)
        elif self.pooler == 'min':
            hidden_states = np.min(hidden_states, axis=1)

        # shape: (n_layers, hidden_dim) or (n_layers, seq_len, hidden_dim)

        return hidden_states


class TransformersVectorizer(Vectorizer):
    # TODO(mm): Support combining embeddings across layers, e.g. taking mean over layer dim

    def __init__(self, model, tokenizer, layer=None, pooler=None, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        assert pooler in [None, 'cls', 'mean', 'max']
        self.pooler = pooler
        self.device = device

    def vectorize_feature(self, feature):
        # TODO(mm): Deprecated. Use vectorie_batch instead
        raise NotImplementedError(
            'Deprecated. Use vectorize_dataset() instead.')

    def vectorize_batch(self, batch, token_level_label_mode, pad_token_id=0, pad_token_label_id=-100):
        hidden_states, inputs = self._get_hidden_states(batch)
        labels = batch[-1]

        if self.layer is not None:
            hidden_states = (hidden_states[self.layer],)

        # apply pooling to get sentence embeddings
        if self.pooler is None:
            # we are not using a pooler
            all_hidden_states = hidden_states

        elif self.pooler == 'cls':
            all_hidden_states = [hidden_state[:, 0, :]
                                 for hidden_state in hidden_states]

        elif self.pooler in ['mean', 'max']:
            all_hidden_states = []
            # Pool only over non-padded hidden states
            # TODO(mm): There is probably a more efficient way to do this without an explicit loop

            for idx, hidden_state in enumerate(hidden_states):
                mask = torch.where(inputs['input_ids'] != pad_token_id, torch.tensor(
                    True).to(self.device), torch.tensor(False).to(self.device))

                pooled_hidden_states = []
                for idx, sequence in enumerate(hidden_state):
                    seq_hidden_states = hidden_state[idx][mask[idx]]
                    if self.pooler == 'mean':
                        seq_hidden_states = torch.mean(
                            seq_hidden_states, dim=0)
                    elif self.pooler == 'max':
                        seq_hidden_states = torch.max(
                            seq_hidden_states, dim=0)[0]
                    pooled_hidden_states.append(seq_hidden_states)

                pooled_hidden_states = torch.stack(pooled_hidden_states, dim=0)
                all_hidden_states.append(pooled_hidden_states)

        # Stack embeddings along first dim (n_layers)
        hidden_states = torch.stack(all_hidden_states, dim=0)

        if hidden_states.size()[0] != 1:  # we have more than one layer
            # Permute tensor to have batch_size first
            hidden_states = hidden_states.permute(1, 0, 2, 3)

        hidden_states = hidden_states.detach().cpu().numpy().squeeze()

        if token_level_label_mode == 'average' and self.pooler is None and self.layer is not None:
            # post-process hidden states. We only do this for token-level tasks
            hidden_states = self._post_process_hidden_states(
                hidden_states, labels, pad_token_label_id=pad_token_label_id)

        return hidden_states

    def _get_hidden_states(self, batch):
        inputs = self._preprocess_batch(batch)

        with torch.no_grad():
            # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
            outputs = self.model(**inputs)
            # hidden_states = list of length num_layers of tensors with shape (bsz, seq_len, hidden_dim)
            hidden_states = outputs[2]

        return hidden_states, inputs

    def _preprocess_batch(self, batch):
        batch = tuple(t.to(self.device) for t in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  }
        return inputs

    def _post_process_hidden_states(self, hidden_states, labels, pad_token_label_id=-100):
        # TODO(mm): Unit test this

        # If token_level_label_mode == 'average' do the following
        #   - label is currently assigned to first subword token, replace it by average of all subword tokens of the word
        #       - collect tokens representation from last true label until -100. These are the ones we will average
        # hidden_states is a batch of hidden states of shape: (bsz, seq_len, hidden_dim)
        # labels is a batch of shape: (bsz, seq_len)

        _hidden_states = np.copy(hidden_states)  # modify copy

        # loop over every sample in the batch
        for sample_idx, sample in enumerate(tqdm(hidden_states, desc='Post-processing hidden states')):
            sample_labels = labels[sample_idx]
            avg_start, avg_end = -1, -1
            # find average start and end positions
            for label_idx, label in enumerate(sample_labels):
                # found the next padded label, special case: CLS token has also pad_token_label_id
                if label == pad_token_label_id and avg_start == -1 and label_idx > 0:
                    avg_start = label_idx - 1
                if avg_start != -1 and label != pad_token_label_id:  # we found the next true label
                    avg_end = label_idx
                    # replace token at position avg_start by average of hidden representations
                    # compute average token representation
                    avg_hidden_state = np.mean(
                        sample[avg_start:avg_end, :], axis=0)

                    _hidden_states[sample_idx, avg_start, :] = avg_hidden_state
                    avg_start, avg_end = -1, -1  # re-init

        return _hidden_states


class GPT2Vectorizer(TransformersVectorizer):
    # TODO(mm): Implement
    pass


class BertVectorizer(TransformersVectorizer):
    def __init__(self, model_type, model_name_or_path, config_name=None, tokenizer_name=None, do_lower_case=False, cache_dir=None, layer=None, pooler=None, device='cpu'):
        if '-cased' in model_name_or_path:
            assert not do_lower_case  # cased models should be cased

        # load pre-trained model and tokenizer
        model, _, tokenizer = load_pretrained_model_and_tokenizer(
            model_type, model_name_or_path,  config_name, tokenizer_name, do_lower_case, output_hidden_states=True, output_attentions=False, cache_dir=cache_dir)
        model.to(device)  # put model on device
        model.eval()  # put model in eval mode

        super().__init__(model=model, tokenizer=tokenizer,
                         layer=layer, pooler=pooler, device=device)


class BertMLMVectorizer(TransformersVectorizer):
    def __init__(self, model_name_or_path, config_name=None, tokenizer_name=None, do_lower_case=False, cache_dir=None, layer=None, pooler=None, device='cpu'):
        if '-cased' in model_name_or_path:
            assert not do_lower_case  # cased models should be cased

        # load pre-trained model and tokenizer
        model, _, tokenizer = load_pretrained_model_and_tokenizer(
            'bert-mlm', model_name_or_path,  config_name, tokenizer_name, do_lower_case, output_hidden_states=True, output_attentions=False, cache_dir=cache_dir)
        model.to(device)  # put model on device
        model.eval()  # put model in eval mode

        super().__init__(model=model, tokenizer=tokenizer,
                         layer=layer, pooler=pooler, device=device)

    def _get_hidden_states(self, batch):
        inputs = self._preprocess_batch(batch)

        encoder = self.model.bert
        mlm = self.model.cls.predictions.transform

        with torch.no_grad():
            # Get hidden states from encoder first
            # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
            outputs = encoder(**inputs)
            # hidden_states = list of length num_layers of tensors with shape (bsz, seq_len, hidden_dim)
            sequence_output, hidden_states = outputs[0], outputs[2]

            # Get hidden states from MLM head (BertPredictionHeadTransform part)
            mlm_hidden_states = mlm(sequence_output)
            hidden_states += (mlm_hidden_states, )

        return hidden_states, inputs


class RobertaVectorizer(TransformersVectorizer):
    def __init__(self, model_type, model_name_or_path,  config_name=None, tokenizer_name=None, do_lower_case=False, cache_dir=None, layer=None, pooler=None, device='cpu'):
        assert not do_lower_case

        # load pre-trained model and tokenizer
        model, _, tokenizer = load_pretrained_model_and_tokenizer(
            model_type, model_name_or_path, config_name, tokenizer_name, do_lower_case, output_hidden_states=True, output_attentions=False, cache_dir=cache_dir)
        model.to(device)  # put model on device
        model.eval()  # put model in eval mode

        super().__init__(model=model, tokenizer=tokenizer,
                         layer=layer, pooler=pooler, device=device)

    def _preprocess_batch(self, batch):
        batch = tuple(t.to(self.device) for t in batch)

        # Pre-trained RoBERTa did not use token_type_ids. The embeddings are available however, fine-tuned sequence-pair models can use them
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  }

        return inputs


class RobertaMLMVectorizer(TransformersVectorizer):
    def __init__(self, model_name_or_path,  config_name=None, tokenizer_name=None, do_lower_case=False, cache_dir=None, layer=None, pooler=None, device='cpu'):
        assert not do_lower_case  # RoBERTa is a cased model

        # load pre-trained model and tokenizer
        model, _, tokenizer = load_pretrained_model_and_tokenizer(
            'roberta-mlm', model_name_or_path, config_name, tokenizer_name, do_lower_case, output_hidden_states=True, output_attentions=False, cache_dir=cache_dir)
        model.to(device)  # put model on device
        model.eval()  # put model in eval mode

        super().__init__(model=model, tokenizer=tokenizer,
                         layer=layer, pooler=pooler, device=device)

    def _preprocess_batch(self, batch):
        batch = tuple(t.to(self.device) for t in batch)

        # Pre-trained RoBERTa did not use token_type_ids. The embeddings are available however, fine-tuned sequence-pair models can use them
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  }

        return inputs

    def _get_hidden_states(self, batch):
        inputs = self._preprocess_batch(batch)

        encoder = self.model.roberta
        mlm = self.model.lm_head

        with torch.no_grad():
            # Get hidden states from encoder first
            # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
            outputs = encoder(**inputs)
            # hidden_states = list of length num_layers of tensors with shape (bsz, seq_len, hidden_dim)
            sequence_output, hidden_states = outputs[0], outputs[2]

            # Get hidden states from MLM head (RobertaLMHead part)
            # We have to mimic the individual steps
            mlm_hidden_states = mlm.dense(sequence_output)
            mlm_hidden_states = F.gelu(mlm_hidden_states)
            mlm_hidden_states = mlm.layer_norm(mlm_hidden_states)
            hidden_states += (mlm_hidden_states, )

        return hidden_states, inputs


class AlbertVectorizer(TransformersVectorizer):
    def __init__(self, model_type, model_name_or_path,  config_name=None, tokenizer_name=None, do_lower_case=True, cache_dir=None, layer=None, pooler=None, device='cpu'):
        assert do_lower_case  # ALBERT is an uncased model

        # load pre-trained model and tokenizer
        model, _, tokenizer = load_pretrained_model_and_tokenizer(
            model_type, model_name_or_path,  config_name, tokenizer_name, do_lower_case, output_hidden_states=True, output_attentions=False, cache_dir=cache_dir)
        model.to(device)  # put model on device
        model.eval()  # put model in eval mode

        super().__init__(model=model, tokenizer=tokenizer,
                         layer=layer, pooler=pooler, device=device)


class AlbertMLMVectorizer(TransformersVectorizer):
    def __init__(self, model_type, model_name_or_path,  config_name=None, tokenizer_name=None, do_lower_case=True, cache_dir=None, layer=None, pooler=None, device='cpu'):
        assert do_lower_case  # ALBERT is an uncased model

        # load pre-trained model and tokenizer
        model, _, tokenizer = load_pretrained_model_and_tokenizer(
            'albert-mlm', model_name_or_path,  config_name, tokenizer_name, do_lower_case, output_hidden_states=True, output_attentions=False, cache_dir=cache_dir)
        model.to(device)  # put model on device
        model.eval()  # put model in eval mode

        super().__init__(model=model, tokenizer=tokenizer,
                         layer=layer, pooler=pooler, device=device)


    def _get_hidden_states(self, batch):
        inputs = self._preprocess_batch(batch)

        encoder = self.model.albert
        mlm = self.model.predictions

        with torch.no_grad():
            # Get hidden states from encoder first
            # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
            outputs = encoder(**inputs)
            # hidden_states = list of length num_layers of tensors with shape (bsz, seq_len, hidden_dim)
            sequence_output, hidden_states = outputs[0], outputs[2]

            # Get hidden states from MLM head (AlbertLMHead part)
            # We have to mimic the individual steps
            mlm_hidden_states = mlm.dense(sequence_output)
            mlm_hidden_states = mlm.activation(mlm_hidden_states)
            mlm_hidden_states = mlm.LayerNorm(mlm_hidden_states) # mlm_hidden_state will have embed_dim instead of hidden_dim (i.e. 128 instead of 768)

            # TODO(mm): pad hidden state with zeros to keep dim consistent (768). Make sure to ignore zeros later on in probing classifier

            hidden_states += (mlm_hidden_states, )

        return hidden_states, inputs


class DistilbertVectorizer(TransformersVectorizer):
    def __init__(self, model_type, model_name_or_path,  config_name=None, tokenizer_name=None, do_lower_case=False, cache_dir=None, layer=None, pooler=None, device='cpu'):
        if '-cased' in model_name_or_path:
            assert not do_lower_case  # cased models should be cased

        # load pre-trained model and tokenizer
        model, _, tokenizer = load_pretrained_model_and_tokenizer(
            model_type, model_name_or_path, config_name, tokenizer_name, do_lower_case, output_hidden_states=True, output_attentions=False, cache_dir=cache_dir)
        model.to(device)  # put model on device
        model.eval()  # put model in eval mode

        super().__init__(model=model, tokenizer=tokenizer,
                         layer=layer, pooler=pooler, device=device)

    def _preprocess_batch(self, batch):
        batch = tuple(t.to(self.device) for t in batch)

        # Distilbert does not use token_type_ids
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  }

        return inputs


class XlnetVectorizer(TransformersVectorizer):
    # TODO(mm): Implement
    pass


class T5Vectorizer(TransformersVectorizer):
    # TODO(mm): Implement
    pass


class ElectraVectorizer(TransformersVectorizer):
    def __init__(self, model_name_or_path, config_name=None, tokenizer_name=None, do_lower_case=False, cache_dir=None, layer=None, pooler=None, device='cpu'):
        # load pre-trained model and tokenizer
        model, _, tokenizer = load_pretrained_model_and_tokenizer(
            'electra', model_name_or_path,  config_name, tokenizer_name, do_lower_case, output_hidden_states=True, output_attentions=False, cache_dir=cache_dir)
        model.to(device)  # put model on device
        model.eval()  # put model in eval mode

        super().__init__(model=model, tokenizer=tokenizer,
                         layer=layer, pooler=pooler, device=device)
