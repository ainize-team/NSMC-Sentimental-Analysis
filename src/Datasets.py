import pandas as pd
import torch

from torch.utils.data import Dataset


class NSMCDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        df = pd.read_csv(csv_file, sep='\t')
        # NaN 값 제거
        df = df.dropna(axis=0)
        # 중복 제거
        df.drop_duplicates(subset=['document'], inplace=True)
        self.input_ids = tokenizer.batch_encode_plus(
            df['document'].to_list(),
            return_tensors='pt',
            truncation=True,
            padding='longest',
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=False,
        )['input_ids']
        self.labels = torch.LongTensor(df['label'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]
