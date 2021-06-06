import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# Models
from transformers import (
    BertForSequenceClassification,  # beomi/kcbert-large
    ElectraForSequenceClassification,  # beomi/KcELECTRA-base
    GPT2ForSequenceClassification,  # skt/kogpt2-base-v2

    PreTrainedTokenizerFast
)

MODEL_FOR_SEQUENCE_CLASSIFICATION = {
    'bert': BertForSequenceClassification,
    'electra': ElectraForSequenceClassification,
    'gpt2': GPT2ForSequenceClassification,
}

TOKENIZER_CLASSES = {
    'bert': PreTrainedTokenizerFast,
    'electra': PreTrainedTokenizerFast,
    'gpt2': PreTrainedTokenizerFast,
}


class NSMCDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.input_ids = []
        for document in df['document']:
            self.input_ids.append(tokenizer(
                document,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt',
                return_token_type_ids=False,
                return_attention_mask=False,
                truncation=True)['input_ids'][0])
        self.labels = torch.LongTensor(df['label'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


def accuracy_score(labels, predicts):
    return (labels == predicts).mean()


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def get_dataloader(dataset, batch_size, shuffle):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle,
    )
    return dataloader
