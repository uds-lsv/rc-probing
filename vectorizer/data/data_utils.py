import csv
import copy
import sys
import json

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

from transformers import (BertTokenizer,
                          DistilBertTokenizer,
                          RobertaTokenizer,
                          AlbertTokenizer,
                          GPT2Tokenizer,
                          XLNetTokenizer,
                          T5Tokenizer,
                          )


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None, text_a_labels=None, text_b_labels=None):
        # TODO(mm): In case of token level tasks there is no text_b. So we can just use label?
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.text_a_labels = text_a_labels
        self.text_b_labels = text_b_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(self, tokens, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.tokens = tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        # label is either a single label in the case of sequence classification or a tensor of labels in the case of token classification
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def create_features_from_examples(examples, processor, tokenizer, max_length=128,
                                  token_level_labels=False, token_level_label_mode='first', labels_file=None, add_special_tokens=True,
                                  cls_token_at_end=False, cls_token="[CLS]",
                                  cls_token_segment_id=1, sep_token="[SEP]", sep_token_extra=False,
                                  pad_on_left=False, pad_token_id=0, pad_token_segment_id=0, pad_token_label_id=-100,
                                  sequence_a_segment_id=0, sequence_b_segment_id=1, mask_padding_with_zero=True, verbose=False):

    if token_level_labels:
        assert token_level_label_mode in ['first', 'all', 'average']
        # token_level_label_mode:
        #   - if first: assign label to first subword token, pad_token_label_id to rest
        #   - if all: assign label to all subword tokens
        #   - if average: average subword vectors into single vector, assing label to it. Here, same as first

        if labels_file is not None:
            label_list = processor.get_labels_from_file(labels_file)
        else:
            label_list = processor.get_labels()
    else:
        label_list = processor.get_labels()
    output_mode = processor.get_output_mode()

    if verbose:
        print(f"Converting {len(examples)} examples to features")
        print("Using label list:", label_list)
        print("Using output mode:", output_mode)

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    for (ex_index, example) in enumerate(tqdm(examples, desc='Creating features')):
        if token_level_labels:
            tokens, labels = [[], []], [[], []]

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2

            # tokenize example word by word and numericalize tokens and corresponding labels (if available)
            # start with text_a
            for word, label in zip(example.text_a, example.text_a_labels):
                word_tokens = tokenizer.tokenize(word)

                if len(tokens[0]) + len(word_tokens) > max_length - special_tokens_count:
                    break  # exceeding max_length. Stop here

                tokens[0].extend(word_tokens)

                if token_level_label_mode in ['first', 'average']:
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    labels[0].extend(
                        [label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                elif token_level_label_mode == 'all':
                    # Use the real label id for the all tokens of the word
                    labels[0].extend([label_map[label]] * (len(word_tokens)))

            # continue with text_b
            if example.text_b is not None:
                for word, label in zip(example.text_b, example.text_b_labels):
                    word_tokens = tokenizer.tokenize(word)

                    if len(tokens[0]) + len(tokens[1]) + len(word_tokens) > max_length - special_tokens_count:
                        break  # exceeding max_length. Stop here

                    if token_level_label_mode in ['first', 'average']:
                        # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                        labels[1].extend([label_map[label]] +
                                         [pad_token_label_id] * (len(word_tokens) - 1))
                    elif token_level_label_mode == 'all':
                        # Use the real label id for the all tokens of the word
                        labels[1].extend([label_map[label]]
                                         * (len(word_tokens)))

            # Combine pair into single sequence:

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.

            if sep_token_extra and len(tokens[1]) > 0:
                # roberta uses an additional separator between pairs of sentences
                sep_token_list = [sep_token, sep_token]
                pad_token_label_id_list = [
                    pad_token_label_id, pad_token_label_id]
            else:
                sep_token_list = [sep_token]
                pad_token_label_id_list = [pad_token_label_id]

            _tokens = tokens  # keep a copy

            # combine text_a and text_b tokens
            if len(tokens[1]) > 0:
                tokens = tokens[0] + sep_token_list + tokens[1] + [sep_token]
                labels = labels[0] + pad_token_label_id_list + \
                    labels[1] + [pad_token_label_id]
                segment_ids = [sequence_a_segment_id] * (len(_tokens[0]) + len(sep_token_list)) + \
                    [sequence_b_segment_id] * (len(_tokens[1]) + 1)
            else:
                tokens = tokens[0] + sep_token_list
                labels = labels[0] + pad_token_label_id_list
                segment_ids = [sequence_a_segment_id] * \
                    (len(_tokens[0]) + len(sep_token_list))

            if cls_token_at_end:
                tokens += [cls_token]
                labels += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]

            else:  # cls_token_at_beginning
                tokens = [cls_token] + tokens
                labels = [pad_token_label_id] + labels
                segment_ids = [cls_token_segment_id] + segment_ids

            # Numericalize tokens
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            token_type_ids = segment_ids

        else:  # sequence-level labels
            inputs = tokenizer.encode_plus(
                example.text_a,
                example.text_b,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
            )

            # token_type_ids indicate if tokens belong to first or second segment ([CLS] [SEG0] [SEP] [SEG1] [SEP])
            input_ids = inputs["input_ids"]
            if isinstance(tokenizer, (RobertaTokenizer, DistilBertTokenizer)):
                # RoBERTa is not assigning token_type_ids by default
                # Distilbert is not using token_type_ids at all
                token_type_ids = [sequence_a_segment_id] * len(input_ids)
            else:
                token_type_ids = inputs["token_type_ids"]

            # Convert sequence label
            if output_mode in ["multi-class-classification", "binary-classification"]:
                label = label_map[example.label]
            elif output_mode == "regression":
                label = float(example.label)
            else:
                raise KeyError(f"Unknown output_mode: {output_mode}")

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)

        if pad_on_left:
            input_ids = ([pad_token_id] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1]
                              * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] *
                              padding_length) + token_type_ids

            if token_level_labels:
                labels = ([pad_token_label_id] * padding_length) + labels

        else:
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + \
                ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + \
                ([pad_token_segment_id] * padding_length)

            if token_level_labels:
                labels += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
            len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length)

        if token_level_labels:
            assert len(labels) == max_length, "Error with input length {} vs {}".format(
                len(labels), max_length)

        # Get tokenizied input corresponding to padded input_ids
        tokens = [tokenizer._convert_id_to_token(
            input_id) for input_id in input_ids]

        if ex_index < 3:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("example text_a: %s" % (example.text_a))
            if example.text_b is not None:
                print("example text_b: %s" % (example.text_b))
            print("tokens: %s" %
                  " ".join([str(x) for x in tokens]))
            print("input_ids: %s" %
                  " ".join([str(x) for x in input_ids]))
            print("attention_mask: %s" %
                  " ".join([str(x) for x in attention_mask]))
            print("token_type_ids: %s" %
                  " ".join([str(x) for x in token_type_ids]))

            if token_level_labels:
                print("labels: %s" %
                      " ".join([str(x) for x in labels]))
            else:
                print("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                tokens=tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=labels if token_level_labels else label)
        )

    return features


def create_dataloader_from_features(features, batch_size):
    print('Creating dataloader from features')

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids)

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataloader


def save_labels_as_npy(features, token_level_labels, output_file):
    labels = None
    for feature in tqdm(features, desc='Saving labels'):
        if token_level_labels:
            label = np.asarray(feature.label)
        else:  # sequence-level label
            label = feature.label
            label = np.asarray([label])

        label = label.reshape(1, -1)
        if labels is None:
            labels = label
        else:
            labels = np.append(labels, label, axis=0)

    with open(output_file, 'wb') as f:
        np.save(f, labels, allow_pickle=True)
        print(f'Saved labels {labels.shape}')


def save_masks_as_npy(features, output_file):
    masks = None
    for feature in tqdm(features, desc='Saving masks'):

        mask = np.asarray(feature.attention_mask).reshape(1, -1)

        if masks is None:
            masks = mask
        else:
            masks = np.append(masks, mask, axis=0)

    with open(output_file, 'wb') as f:
        np.save(f, masks, allow_pickle=True)
        print(f'Saved attention masks {masks.shape}')


def read_labels_from_file(file_path):
    assert file_path is not None
    assert file_path.endswith('.txt')

    with open(file_path, "r") as f:
        labels = f.read().splitlines()

    return labels


def read_token_level_task_examples_from_file(file_path, sample_delim='*', text_b_delim='+', word_col=1, label_col=None, split='train'):
    # This works for our spacy annotated file format and .conllu format.

    assert label_col is not None

    sample_idx = 0
    examples = []

    with open(file_path, encoding="utf-8") as f:
        # list of lists. words of text_a will be at position 0, words of text_b at position 1 (empty list if there is no text_b)
        # same for ids and labels
        words, labels = [[], []], [[], []]
        text_idx = 0

        for line in tqdm(f, desc='Creating examples'):
            # line contains a comment or sentence level label
            if line.startswith('#') or line.startswith('label'):
                continue  # ignore line
            elif line.startswith(text_b_delim):  # text_b follows
                text_idx = 1
            elif line.startswith(sample_delim):  # next sample follows
                # create InputExample from collected words and labels
                example = InputExample(
                    guid="{}-{}".format(split, sample_idx),
                    text_a=words[0], text_a_labels=labels[0],
                    text_b=words[1] if len(words[1]) > 0 else None,
                    text_b_labels=labels[1] if len(labels[1]) > 0 else None
                )
                examples.append(example)
                sample_idx += 1  # increment example index & re-initialize
                words, labels = [[], []], [[], []]
                text_idx = 0
                # if sample_idx < 3:
                #     print(example)

            else:  # line contains word and corresponding labels
                splits = line.split("\t")
                # make sure we have at least a word and a label
                assert len(splits) >= 2

                # collect word and labels
                words[text_idx].append(splits[word_col].strip())
                labels[text_idx].append(splits[label_col].strip())

        if words:  # last sample
            example = InputExample(
                guid="{}-{}".format(split, sample_idx),
                text_a=words[0], text_a_labels=labels[0],
                text_b=words[1] if len(words[1]) > 0 else None,
                text_b_labels=labels[1] if len(labels[1]) > 0 else None
            )
            examples.append(example)

    return examples
