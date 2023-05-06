import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
import os
import csv
import torch 


GlossSelectionRecord = namedtuple("GlossSelectionRecord", [
                                  "id", "sentence", "sense_keys", "glosses", "target_words"])
BertInput = namedtuple(
    "BertInput", ["input_ids", "input_mask", "segment_ids", "label_id"])


def load_dataset(csv_path, tokenizer, max_sequence_length):
    def deserialize_csv_record(row):
        return GlossSelectionRecord(row[0], row[1], eval(row[2]), eval(row[3]), [int(t) for t in eval(row[4])])

    return _load_and_cache_dataset(
        csv_path,
        tokenizer,
        max_sequence_length,
        deserialize_csv_record
    )


def collate_batch(batch):
    max_seq_length = len(batch[0][0].input_ids)

    collated = []
    for sub_batch in batch:
        batch_size = len(sub_batch)
        sub_collated = [torch.zeros([batch_size, max_seq_length], dtype=torch.long) for _ in range(3)] + [torch.zeros([batch_size], dtype=torch.long)]

        for i, bert_input in enumerate(sub_batch):
            sub_collated[0][i] = torch.tensor(
                bert_input.input_ids, dtype=torch.long)
            sub_collated[1][i] = torch.tensor(
                bert_input.input_mask, dtype=torch.long)
            sub_collated[2][i] = torch.tensor(
                bert_input.segment_ids, dtype=torch.long)
            sub_collated[3][i] = torch.tensor(
                bert_input.label_id, dtype=torch.long)

        collated.append(sub_collated)

    return collated


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _load_and_cache_dataset(csv_path, tokenizer, max_sequence_length, deserialze_fn):
    data_dir = os.path.dirname(csv_path)
    dataset_name = os.path.basename(csv_path).split('.')[0]
    cached_features_file = os.path.join(
        data_dir, f"cached_{dataset_name}_{max_sequence_length}")
    if os.path.exists(cached_features_file):
        print(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        print(f"Creating features from dataset {csv_path}")
        records = _create_records_from_csv(csv_path, deserialze_fn)

        features = _create_features_from_records(records, max_sequence_length, tokenizer,
                                                 cls_token=tokenizer.cls_token,
                                                 sep_token=tokenizer.sep_token,
                                                 cls_token_segment_id=1,
                                                 pad_token_segment_id=0)

    # print("Saving features into cached file %s", cached_features_file)
    # torch.save(features, cached_features_file)

    class FeatureDataset(torch.utils.data.Dataset):
        def __init__(self, features_):
            self.features = features_

        def __getitem__(self, index):
            return self.features[index]

        def __len__(self):
            return len(self.features)

    return FeatureDataset(features)


def _create_records_from_csv(csv_path, deserialize_fn):
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # read off header

        return [deserialize_fn(row) for row in reader]


def _create_features_from_records(records, max_seq_length, tokenizer, cls_token_at_end=False, pad_on_left=False,
                                  cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                  sequence_a_segment_id=0, sequence_b_segment_id=1,
                                  cls_token_segment_id=1, pad_token_segment_id=0,
                                  mask_padding_with_zero=True, disable_progress_bar=False):
    features = []
    for record in tqdm(records):
        tokens_a = tokenizer.tokenize(record.sentence)

        sequences = [(gloss, 1 if i in record.target_words else 0)
                     for i, gloss in enumerate(record.glosses)]

        pairs = []
        for seq, label in sequences:
            tokens_b = tokenizer.tokenize(seq)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            tokens = tokens_a + [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] *padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            pairs.append(
                BertInput(input_ids=input_ids, input_mask=input_mask,
                          segment_ids=segment_ids, label_id=label)
            )
        features.append(pairs)

    return features


