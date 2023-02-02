import argparse
import os
import logging

from vectorizer.data.processors import processors
from vectorizer.data.data_utils import create_features_from_examples, save_labels_as_npy, save_masks_as_npy, create_dataloader_from_features

from vectorizer.vectorizers import (GloveVectorizer, FastTextVectorizer, FlairEmbeddingsVectorizer,
                                    ElmoVectorizer,
                                    BertVectorizer, BertMLMVectorizer,
                                    RobertaVectorizer, RobertaMLMVectorizer,
                                    AlbertVectorizer,
                                    ElectraVectorizer)

from vectorizer.indexer import index_features
from vectorizer.configs.config_utils import read_config

from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)


def _create_embddings_path(config):
    # Create dir if it doesn't exist yet
    if not os.path.exists(config.vectorizer.embeddings_dir):
        os.makedirs(config.vectorizer.embeddings_dir)

    # For fine-tuned models we also collect the name of the downstream task
    if 'checkpoints' in config.model.model_name_or_path:
        assert '-finetuned' in config.model.model_type

        if '-adapters' in config.model.model_name_or_path:
            encoder_name = 'adapter-'
            encoder_name += config.model.model_name_or_path.split(
                '/')[-1].split('_')[-2]
            downstream_task = config.model.model_name_or_path.split(
                '/')[2].split('-')[0]
        elif '-pooling' in config.model.model_name_or_path:
            pooler = config.model.model_type.split('-')[0]
            encoder_name = f'{pooler}-'
            encoder_name += config.model.model_name_or_path.split(
                '/')[-1].split('_')[-2]
            downstream_task = config.model.model_name_or_path.split(
                '/')[2].split('-')[0]
        else:
            encoder_name = config.model.model_name_or_path.split(
                '/')[-1].split('_')[-2]
            downstream_task = config.model.model_name_or_path.split('/')[2]

        name = f"{config.input.input_file_name}_{encoder_name}-finetuned-{downstream_task}_layer={config.model.layer}_pooler={config.model.pooler}_{config.input.max_length}.hdf5"
    else:
        encoder_name = config.model.model_name_or_path.split('/')[-1]
        if config.input.task_type == 'token-level':
            name = f"{config.input.input_file_name}_{encoder_name}_layer={config.model.layer}_{config.input.max_length}_labelmode={config.input.token_level_label_mode}.hdf5"
        else:
            name = f"{config.input.input_file_name}_{encoder_name}_layer={config.model.layer}_pooler={config.model.pooler}_{config.input.max_length}.hdf5"

    return os.path.join(config.vectorizer.embeddings_dir, name)


def main(args, config):
    # Create data processor and get examples
    processor = processors[config.input.dataset]()

    input_file = os.path.join(
        config.input.input_dir, config.input.input_file_name)

    split = 'train' if 'train' in input_file else 'dev'
    examples = processor.get_examples(file_name=input_file, split=split)

    if config.model.model_type == 'elmo':
        cuda_device = 0 if args.cuda else -1
        vectorizer = ElmoVectorizer(
            layer=config.model.layer, pooler=config.model.pooler, cuda_device=cuda_device)

    elif config.model.model_type == 'glove':
        vectorizer = GloveVectorizer(pooler=config.model.pooler)
        config.model.layer = 0  # there's only a single layer

    elif config.model.model_type == 'fasttext':
        vectorizer = FastTextVectorizer(pooler=config.model.pooler)
        config.model.layer = 0  # there's only a single layer

    elif config.model.model_type == 'flair':
        vectorizer = FlairEmbeddingsVectorizer(config.model.pooler)
        config.model.layer = 0  # there's only a single layer

    else:
        # TODO(mm): Refactor, less copying.

        # Create vectorizer
        if config.model.model_type in ['bert', 'bert-finetuned', 'bert-finetuned-squad', 'adapter-bert-finetuned']:
            vectorizer = BertVectorizer(
                config.model.model_type, config.model.model_name_or_path,
                config_name=config.model.config_name if config.model.config_name else config.model.model_name_or_path,
                tokenizer_name=config.model.tokenizer_name if config.model.tokenizer_name else config.model.model_name_or_path,
                do_lower_case=config.model.do_lower_case,
                cache_dir=config.model.cache_dir, layer=config.model.layer, pooler=config.model.pooler,
                device='cuda' if args.cuda else 'cpu')
            tokenizer = vectorizer.tokenizer  # get tokenizer

        elif config.model.model_type == 'bert-mlm':
            vectorizer = BertMLMVectorizer(
                config.model.model_name_or_path,
                config_name=config.model.config_name if config.model.config_name else config.model.model_name_or_path,
                tokenizer_name=config.model.tokenizer_name if config.model.tokenizer_name else config.model.model_name_or_path,
                do_lower_case=config.model.do_lower_case,
                cache_dir=config.model.cache_dir, layer=config.model.layer, pooler=config.model.pooler,
                device='cuda' if args.cuda else 'cpu')
            tokenizer = vectorizer.tokenizer  # get tokenizer

        elif config.model.model_type in ['albert', 'albert-finetuned', 'albert-finetuned-squad', 'adapter-albert-finetuned']:
            vectorizer = AlbertVectorizer(
                config.model.model_type, config.model.model_name_or_path,
                config_name=config.model.config_name if config.model.config_name else config.model.model_name_or_path,
                tokenizer_name=config.model.tokenizer_name if config.model.tokenizer_name else config.model.model_name_or_path,
                do_lower_case=config.model.do_lower_case,
                cache_dir=config.model.cache_dir, layer=config.model.layer, pooler=config.model.pooler,
                device='cuda' if args.cuda else 'cpu')
            tokenizer = vectorizer.tokenizer

        elif config.model.model_type in ['roberta', 'roberta-finetuned', 'roberta-finetuned-squad', 'adapter-roberta-finetuned', 'meanpooling-roberta-finetuned', 'fivepooling-roberta-finetuned']:
            vectorizer = RobertaVectorizer(
                config.model.model_type, config.model.model_name_or_path,
                config_name=config.model.config_name if config.model.config_name else config.model.model_name_or_path,
                tokenizer_name=config.model.tokenizer_name if config.model.tokenizer_name else config.model.model_name_or_path,
                do_lower_case=config.model.do_lower_case,
                cache_dir=config.model.cache_dir, layer=config.model.layer, pooler=config.model.pooler,
                device='cuda' if args.cuda else 'cpu')
            tokenizer = vectorizer.tokenizer

        elif config.model.model_type == 'roberta-mlm':
            vectorizer = RobertaMLMVectorizer(
                config.model.model_name_or_path,
                config_name=config.model.config_name if config.model.config_name else config.model.model_name_or_path,
                tokenizer_name=config.model.tokenizer_name if config.model.tokenizer_name else config.model.model_name_or_path,
                do_lower_case=config.model.do_lower_case,
                cache_dir=config.model.cache_dir, layer=config.model.layer, pooler=config.model.pooler,
                device='cuda' if args.cuda else 'cpu')
            tokenizer = vectorizer.tokenizer

        elif config.model.model_type == 'electra':
            raise NotImplementedError("Electra is not yet integrated.")

        else:
            raise NotImplementedError(
                f"Unknown model_type: {config.model.model_type}")

        # Convert examples into features
        features = create_features_from_examples(examples, processor, tokenizer, max_length=config.input.max_length,
                                                 token_level_labels=True if config.input.task_type == 'token-level' else False,
                                                 token_level_label_mode=config.input.token_level_label_mode,
                                                 labels_file=os.path.join(
                                                     config.input.input_dir, config.input.labels_file_name) if config.input.labels_file_name else None,
                                                 add_special_tokens=True,
                                                 cls_token_at_end=False, cls_token=tokenizer.cls_token,
                                                 cls_token_segment_id=2 if config.model.model_type in [
                                                     "xlnet"] else 0,
                                                 sep_token=tokenizer.sep_token, sep_token_extra=bool(
                                                     config.model.model_type in ["roberta"]),
                                                 pad_on_left=bool(
                                                     config.model.model_type in ["xlnet"]),
                                                 pad_token_id=tokenizer.convert_tokens_to_ids(
                                                     [tokenizer.pad_token])[0],
                                                 pad_token_segment_id=4 if config.model.model_type in [
                                                     "xlnet"] else 0,
                                                 pad_token_label_id=CrossEntropyLoss().ignore_index,
                                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                                 mask_padding_with_zero=True, verbose=False)

        assert len(examples) == len(features)

        if config.labels.enable:
            # Create dir if it doesn't exist yet
            if not os.path.exists(config.labels.labels_dir):
                os.makedirs(config.labels.labels_dir)

            if config.input.task_type == 'token-level':
                labels_output_file = os.path.join(
                    config.labels.labels_dir, f"{config.input.input_file_name}_{config.input.task_type}_{config.input.max_length}_labelmode={config.input.token_level_label_mode}_{config.input.dataset}_labels.npy")
            else:
                labels_output_file = os.path.join(
                    config.labels.labels_dir, f"{config.input.input_file_name}_{config.input.task_type}_{config.input.dataset}_labels.npy")

            # create lables only if the file does not exist yet
            if not os.path.exists(labels_output_file):
                save_labels_as_npy(features, token_level_labels=True if config.input.task_type ==
                                   'token-level' else False, output_file=labels_output_file)
            else:
                print(f'Labels file already exist: {labels_output_file}')

            # Save attention masks
            if config.labels.masks_dir is not None:
                # Create dir if it doesn't exist yet
                if not os.path.exists(config.labels.masks_dir):
                    os.makedirs(config.labels.masks_dir)

                if config.input.task_type == 'token-level':
                    masks_output_file = os.path.join(
                        config.labels.masks_dir, f"{config.input.input_file_name}_{config.input.task_type}_{config.input.max_length}_labelmode={config.input.token_level_label_mode}_input-masks.npy")
                else:
                    masks_output_file = os.path.join(
                        config.labels.masks_dir, f"{config.input.input_file_name}_{config.input.task_type}_input-masks.npy")

                # create lables only if the file does not exist yet
                if not os.path.exists(masks_output_file):
                    save_masks_as_npy(features, output_file=masks_output_file)
                else:
                    print(f'Masks file already exist: {masks_output_file}')

        if config.indexer.enable:
            index_features(features, config, sort_index=True)

    if config.vectorizer.enable:
        embeddings_file = _create_embddings_path(config)
        if not os.path.exists(embeddings_file):
            if config.model.model_type in ['glove', 'fasttext', 'flair', 'elmo']:
                # For baseline models and ELMo we vectorize using examples
                embeddings = vectorizer.vectorize_features(examples)
            else:
                # For Transformer models we vectorize in batches using a dataloader
                dataloader = create_dataloader_from_features(
                    features, config.input.batch_size)

                # Create embeddings from dataloader
                embeddings = vectorizer.vectorize_dataset(
                    dataloader, token_level_label_mode=config.input.token_level_label_mode,
                    pad_token_id=tokenizer.convert_tokens_to_ids(
                        [tokenizer.pad_token])[0],
                    pad_token_label_id=CrossEntropyLoss().ignore_index)

            # Save embeddings
            vectorizer.make_hdf5_file_from_embeddings(
                embeddings, output_file=embeddings_file)
        else:
            print(f'Embeddings already exist: {embeddings_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--config", type=str,
                        required=True,
                        help="A config file specifying what to do.")

    parser.add_argument("--layer", type=int,
                        default=None,
                        help="Layer from which to extract embeddings. Overwrites config.model.layer if specified.")

    parser.add_argument("--pooler", type=str,
                        default=None,
                        help="Pooler to use for creating sentence embeddings. Overwrites config.model.pooler if specified.")

    parser.add_argument("--cuda", action='store_true',
                        help='Use this flag to put model and data on GPU.')

    args = parser.parse_args()

    config = read_config(args.config)

    # Overwrite config based on args
    if args.layer is not None:
        config['model']['layer'] = args.layer

    if args.pooler is not None:
        config['model']['pooler'] = args.pooler

    print(f"Config file: {config}")

    # Run some assertions
    if config.model.model_type in ['glove', 'fasttext', 'flair', 'elmo']:
        # Padding is not yet implemented for these models, hence only sentence embeddings are supported
        # TODO(mm): Implement padding for ELMo and static embedding models
        assert config.model.pooler is not None
        assert config.model.task_type not in ['token-level']

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    main(args, config)
