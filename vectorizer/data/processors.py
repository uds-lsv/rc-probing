from tqdm import tqdm

from transformers import DataProcessor
from vectorizer.data.data_utils import InputExample, read_token_level_task_examples_from_file, read_labels_from_file


class SequenceLevelTaskProcessor(DataProcessor):
    """Processor for sequence-level classification tasks."""

    def get_examples(self, file_name, split, has_header=False):
        return self._create_examples(
            self._read_tsv(file_name), split, has_header)

    def get_labels(self):
        # Implement in child class
        raise NotImplementedError()

    def get_output_mode(self):
        # Implement in child class
        raise NotImplementedError()

    def _create_examples(self, lines, split, has_header):
        # Implement in child class
        raise NotImplementedError()


class TokenLevelTaskProcessor(DataProcessor):
    """Processor for token-level classification tasks."""

    def __init__(self, sample_delim='*', text_b_delim='+', word_col=1, label_col=3):
        self.sample_delim = sample_delim
        self.text_b_delim = text_b_delim
        self.word_col = word_col
        self.label_col = label_col

    def get_examples(self, file_name, split):
        # TODO(mm): This assumes a certain file format. Overwrite if using different format.
        return read_token_level_task_examples_from_file(file_name, sample_delim=self.sample_delim,
                                                        text_b_delim=self.text_b_delim, word_col=self.word_col, label_col=self.label_col, split=split)

    def get_labels(self):
        # Implement in child class
        raise NotImplementedError()

    def get_labels_from_file(self, file_path):
        # Create labels from .tsv files using:
        #   cat train.tsv dev.tsv test.tsv | cut -f 2 | grep -v "^$"| sort | uniq > pos_labels.txt
        return read_labels_from_file(file_path)

    def get_output_mode(self):
        # Implement in child class
        raise NotImplementedError()


class RCProcessor(SequenceLevelTaskProcessor):
    """Processor for RC data (sequence classification)."""

    def get_labels(self):
        return ['0', '1']

    def get_output_mode(self):
        return 'binary-classification'

    def _create_examples(self, lines, split, has_header):
        examples = []

        for (i, line) in enumerate(tqdm(lines, desc='Creating examples')):
            if has_header and i == 0:
                continue  # skip header

            guid = f"{split}-{i}"
            text = line[5]  # modified sentence column
            label = line[0]  # label column

            example = InputExample(
                guid=guid, text_a=text, text_b=None, label=label,
                text_a_labels=None, text_b_labels=None)
            examples.append(example)

        return examples



class LinkedWT2Processor(SequenceLevelTaskProcessor):
    """Processor for Linked-Wikitext-2 dataset (preprocessed version, single sentence per line)."""

    def get_labels(self):
        return ['0']

    def get_output_mode(self):
        return 'binary-classification'

    def _create_examples(self, lines, split, has_header):
        examples = []

        for (i, line) in enumerate(tqdm(lines, desc='Creating examples')):
            if has_header and i == 0:
                continue  # skip header

            guid = f"{split}-{i}"
            text = line[0]
            label = '0'

            example = InputExample(
                guid=guid, text_a=text, text_b=None, label=label,
                text_a_labels=None, text_b_labels=None)
            examples.append(example)

        return examples


class ColaSpacyUPOS(TokenLevelTaskProcessor):
    def __init__(self, sample_delim='*', text_b_delim='+', word_col=1, label_col=2):
        super().__init__(sample_delim, text_b_delim, word_col=1, label_col=2)

    def get_output_mode(self):
        # Implement in child class
        return 'classification'


class SentEvalPastPresent(SequenceLevelTaskProcessor):
    """Processor for SemEval past-present probing data (sequence classification)."""

    def get_labels(self):
        return ['PAST', 'PRES']

    def get_output_mode(self):
        return 'binary-classification'

    def _create_examples(self, lines, split, has_header):
        examples = []

        for (i, line) in enumerate(tqdm(lines, desc='Creating examples')):
            if has_header and i == 0:
                continue  # skip header

            guid = f"{split}-{i}"
            text = line[0]  # sentence column
            label = line[1]  # label column

            example = InputExample(
                guid=guid, text_a=text, text_b=None, label=label,
                text_a_labels=None, text_b_labels=None)
            examples.append(example)

        return examples


class SentEvalBigramShift(SentEvalPastPresent):
    """Processor for SemEval bigram-shift probing data (sequence classification)."""

    def get_labels(self):
        return ['O', 'I']


class SentEvalOddManOut(SentEvalPastPresent):
    """Processor for SemEval bigram-shift probing data (sequence classification)."""

    def get_labels(self):
        return ['O', 'C']


processors = {
    'rc-probing': RCProcessor,
    'linked-wt2': LinkedWT2Processor,
    'cola-spacy-upos': ColaSpacyUPOS,
    'senteval-past-present': SentEvalPastPresent,
    'senteval-bigram-shift': SentEvalBigramShift,
    'senteval-odd-man-out': SentEvalOddManOut,
    'senteval-coordination-inversion': SentEvalBigramShift,  # it uses the same labels
}
