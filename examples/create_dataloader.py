import argparse

import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler

from vectorizer.utils import load_embeddings_from_hdf5, load_labels_from_npy

# This example shows how to create DataLoader from existing embedding and label files.


def convert_labels_to_tensor(labels):
    # Convert to tensor
    labels = torch.tensor(
        [label for label in labels], dtype=torch.float)

    return labels


def convert_embedddings_to_tensor(embeddings):
    embeddings = np.asarray(embeddings).squeeze()

    # Convert to tensor
    embeddings = torch.tensor(
        [embedding for embedding in embeddings], dtype=torch.float)

    return embeddings


def create_dataset_from_files(embeddings_file, labels_file):
    embeddings = load_embeddings_from_hdf5(embeddings_file)
    labels = load_labels_from_npy(labels_file)

    # print(embeddings.shape)
    # print(embeddings[0])
    # assert False

    # convert to tensor
    embeddings = convert_embedddings_to_tensor(embeddings)
    labels = convert_labels_to_tensor(labels)

    # create dataset
    dataset = TensorDataset(embeddings, labels)

    return dataset


def main(args):
    # Create dataloader
    dataset = create_dataset_from_files(args.hdf5_file, args.labels_file)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler,
                            batch_size=args.batch_size)

    for batch in dataloader:
        inputs = batch[0]
        labels = batch[1]
        print(inputs.shape)
        print(inputs)
        print(labels.shape)
        print(labels)
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--hdf5_file", type=str,
                        # default="/datasets/probing/rc_probing/v0.3/balanced/embeddings/sentence-embeddings/dev.tsv_bert-base-cased_layer=0_pooler=mean.hdf5",
                        # default="/datasets/probing/rc_probing/v0.3/balanced/embeddings/sentence-embeddings/dev.tsv_glove_layer=0_pooler=mean.hdf5",
                        default="/datasets/probing/rc_probing/v0.3/balanced/embeddings/sentence-embeddings/train.tsv_glove_layer=0_pooler=mean.hdf5",
                        help="The input embedding file.")

    parser.add_argument("--labels_file", type=str,
                        default="/datasets/probing/rc_probing/v0.3/balanced/labels/train_labels.npy",
                        help="The input label file.")

    parser.add_argument("--batch_size", type=int,
                        default=10,
                        help="Batch size.")

    args = parser.parse_args()

    main(args)
