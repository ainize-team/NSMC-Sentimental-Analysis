import torch

from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizerFast
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score

from Datasets import NSMCDataset


class KoBERT(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.bert = BertForSequenceClassification.from_pretrained(self.hparams.model_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.hparams.model_path)

    def forward(self, **kwargs):
        return self.bert(**kwargs)

    def step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data, labels=labels)

        loss = output.loss

        y_true = labels.tolist()
        y_pred = output.logits.argmax(dim=-1).tolist()

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def epoch_end(self, outputs, state):
        loss = torch.tensor(0, dtype=torch.float)
        y_true = []
        y_pred = []
        for output in outputs:
            loss += output['loss'].cpu().detach()
            y_true.extend(output['y_true'])
            y_pred.extend(output['y_pred'])
        loss = loss / len(outputs)

        self.log(state + '_loss', float(loss), on_epoch=True, prog_bar=True)
        self.log(state + '_acc', accuracy_score(y_true, y_pred), on_epoch=True, prog_bar=True)

    def training_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='train')

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='val')

    def configure_optimizers(self):
        optimizer = AdamW(self.bert.parameters(), lr=self.hparams.lr)

        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps
        )

        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [lr_scheduler]

    def dataloader(self, file_path, shuffle) -> DataLoader:
        dataset = NSMCDataset(file_path, self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.hparams.train_data_path, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.hparams.val_data_path, shuffle=False)

    def save_hugginface(self):
        self.bert.save_pretrained(self.hparams.save_path)