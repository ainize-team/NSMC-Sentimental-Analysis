import os
import argparse
import logging

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from PLModels import KoBERT


def get_parser():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(description='Fine Tune KoBART for Naver Movie Review Sentiment Analysis')
    parser.add_argument('--train_data_path',
                        default='./nsmc/ratings_train.txt',
                        help='train data file path')
    parser.add_argument('--val_data_path',
                        default='./nsmc/ratings_test.txt',
                        help='validation data file path')
    parser.add_argument('--save_path',
                        default='./huggingface_model',
                        help='model and tokenizer path')
    parser.add_argument('--max_epochs',
                        default=3,
                        type=int,
                        help='')
    parser.add_argument('--model_path',
                        default='beomi/kcbert-base',
                        help='')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='how many samples per batch to load')
    parser.add_argument('--num_workers',
                        default=4,
                        type=int,
                        help='how many subprocesses to use for data loading')
    parser.add_argument('--lr',
                        type=float,
                        default=5e-6,
                        help='The initial learning rate')
    parser.add_argument('--warmup_ratio',
                        type=float,
                        default=0.0,
                        help='warmup ratio')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='16-bit precision')
    return parser


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    parser = get_parser()
    args = vars(parser.parse_args())
    logging.info(args)

    checkpoint_callback = ModelCheckpoint(
        filename='epoch{epoch}-val_acc{val_acc:.4f}',
        monitor='val_acc',
        save_top_k=3,
        mode='max',
        auto_insert_metric_name=False,
    )

    model = KoBERT(**args)
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=args['max_epochs'],
        num_sanity_val_steps=0,
        deterministic=torch.cuda.is_available(),
        gpus=-1 if torch.cuda.is_available() else None,
    )
    trainer.fit(model)
    model.save_hugginface()


if __name__ == '__main__':
    main()
