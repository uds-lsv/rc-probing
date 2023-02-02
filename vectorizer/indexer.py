import copy
import json
import os
import csv
import itertools

from tqdm import tqdm


class Token2SampleIndex():
    def __init__(self):
        self.token_to_sample_indexer = {}
        # ignore padding tokens when building index
        self.tokens_to_ignore = ['[PAD]', '<pad>']
        # TODO(mm): What else should we ignore? Punctuation?

    def add(self, token, sample_idx, token_idx):
        if token not in self.tokens_to_ignore:
            if token in self.token_to_sample_indexer:
                self.token_to_sample_indexer[token].append(
                    [sample_idx, token_idx])
            else:
                self.token_to_sample_indexer[token] = [[sample_idx, token_idx]]

    def remove_rare_words(self, min_freq):
        infrequent_words = list(filter(lambda w: len(
            self.token_to_sample_indexer[w]) < min_freq, self.token_to_sample_indexer.keys()))

        for w in infrequent_words:
            del self.token_to_sample_indexer[w]

    def sort_by_frequency(self):
        self.token_to_sample_indexer = {w: self.token_to_sample_indexer[w] for w in sorted(
            self.token_to_sample_indexer, key=lambda w: len(self.token_to_sample_indexer[w]), reverse=True)}

    def get_frequency(self, token):
        return len(self.token_to_sample_indexer[token])

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.token_to_sample_indexer)
        return output

    def save_as_dict(self, output_file):
        json.dump(self.token_to_sample_indexer,
                  open(output_file, 'w'), indent=1)

    def save_frequencies(self, output_file):
        fieldnames = ['token', 'frequency', 'sentence pairs']
        writer = csv.DictWriter(open(output_file, 'w'), fieldnames=fieldnames)
        writer.writeheader()

        for token in tqdm(self.token_to_sample_indexer, desc='Computing sentence pairs per token'):
            # Compute also number of sentence pairs that contain this token
            # but ingore padding token. There are too many combinations
            index_pairs = []
            indices = range(len(self.token_to_sample_indexer[token]))
            # Get all pairs of sentences that contain token
            index_pairs = [(i, j) for i, j in itertools.product(
                indices, indices) if i != j]

            writer.writerow({'token': token, 'frequency': len(
                self.token_to_sample_indexer[token]), 'sentence pairs': len(index_pairs)})


def _index_tokens(tokens, sample_idx, token_to_sample_indexer):
    for token_idx, token in enumerate(tokens):
        token_to_sample_indexer.add(token, sample_idx, token_idx)


def index_features(features, config, sort_index=True):
    sample_index = 0
    token_to_sample_indexer = Token2SampleIndex()

    for feature in tqdm(features, desc='Indexing tokens'):
        tokens = feature.tokens
        _index_tokens(tokens, sample_index, token_to_sample_indexer)
        sample_index += 1

    # remove words that appear less than min_freq times
    token_to_sample_indexer.remove_rare_words(config.indexer.min_freq)

    # sort index by frequency
    if sort_index:
        token_to_sample_indexer.sort_by_frequency()

    if config.indexer.index_dir is not None:
        # Create dir if it doesn't exist yet
        if not os.path.exists(config.indexer.index_dir):
            os.makedirs(config.indexer.index_dir)

        # Save index file
        index_output_file = os.path.join(
            config.indexer.index_dir, f"{config.input.input_file_name}_{config.model.model_name_or_path.split('/')[-1]}_index.json")
        print(f'Saving token index to: {index_output_file}')
        token_to_sample_indexer.save_as_dict(index_output_file)

        # Save frequencies file
        freq_output_file = os.path.join(
            config.indexer.index_dir, f"{config.input.input_file_name}_{config.model.model_name_or_path.split('/')[-1]}_freq.tsv")
        print(f'Saving token frequency to: {freq_output_file}')
        token_to_sample_indexer.save_frequencies(freq_output_file)

    return token_to_sample_indexer
