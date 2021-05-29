import argparse
import datetime
import gc

import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from firebase_utils import init_firebase, download_model, download_data, upload_model
from utils import MODEL_FOR_SEQUENCE_CLASSIFICATION, TOKENIZER_CLASSES
from utils import NSMCDataset, get_dataloader, set_seed, accuracy_score


def get_args():
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser(
        description='Fine Tune PLM(at huggingface) for Naver Movie Review Sentiment Analysis')
    parser.add_argument('--model_name',
                        required=True,
                        type=str,
                        help='base_model(skt-kogpt2-base-v2, beomi-KcELECTRA-base, beomi-kcbert-large)')
    parser.add_argument('--model_version',
                        type=str,
                        default='2021_05_29')
    parser.add_argument('--model_type',
                        required=True,
                        type=str,
                        help='model_type(gpt2, electra, bert)')
    parser.add_argument('--max_seq_len',
                        default=128,
                        type=int,
                        help='')
    parser.add_argument('--num_train_epochs',
                        default=10,
                        type=int,
                        help='')
    parser.add_argument('--warmup_proportion',
                        default=0.0,
                        type=float,
                        help='')
    parser.add_argument('--learning_rate',
                        default=5e-5,
                        type=float,
                        help='')
    parser.add_argument('--seed',
                        default=42,
                        type=int,
                        help='', )
    parser.add_argument('--batch_size',
                        default=32,
                        type=int,
                        help='')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='if value is passed, showing progress bar for each epoch')
    parser.add_argument('--output_version',
                        default=f'{now.year}_{now.month}_{now.day}',
                        type=str)
    return parser.parse_args()


def train(model, train_dataloader, train_sampler, device, is_master, args):
    model.train()
    global_total_step = len(train_dataloader) * args.num_train_epochs
    global_step = 0
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=global_total_step * args.warmup_proportion,
                                                num_training_steps=global_total_step)
    with tqdm(total=global_total_step, unit='step', disable=not is_master or not args.verbose) as t:
        total = 0
        total_loss = 0
        for epoch in range(args.num_train_epochs):
            if train_sampler:
                train_sampler.set_epoch(epoch)
            for batch in train_dataloader:
                global_step += 1
                b_input_ids = batch[0].to(device, non_blocking=True)
                b_labels = batch[1].to(device, non_blocking=True)
                model.zero_grad(set_to_none=True)
                outputs = model(
                    input_ids=b_input_ids,
                    labels=b_labels
                )
                loss, logits = (outputs['loss'], outputs['logits']) if isinstance(outputs, dict) else (
                    outputs[0], outputs[1])

                loss.backward()
                optimizer.step()
                scheduler.step()

                preds = logits.detach().argmax(dim=-1).cpu().numpy()
                out_label_ids = b_labels.detach().cpu().numpy()

                if is_master:
                    batch_loss = loss.item() * len(b_input_ids)

                    total += len(b_input_ids)
                    total_loss += batch_loss

                    if args.verbose:
                        t.set_postfix(loss='{:.6f}'.format(batch_loss),
                                      accuracy='{:.2f}'.format(accuracy_score(out_label_ids, preds) * 100))
                        t.update(1)
                del b_input_ids
                del outputs
                del loss


def train_single(args):
    # device 를 할당 한다.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Model 과 Tokenizer를 불러온다.
    model = MODEL_FOR_SEQUENCE_CLASSIFICATION[args.model_type].from_pretrained('./model')
    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained('./model')
    # Dataset 을 만든다.
    train_dataset = NSMCDataset(args.train_df, tokenizer, args.max_seq_len)
    tokenizer.save_pretrained('./output_dir', legacy_format=False)
    model.to(device)
    batch_size = args.batch_size
    while batch_size:
        try:
            train_dataloader = get_dataloader(train_dataset, batch_size, shuffle=True)
            train(model, train_dataloader, None, device, True, args)
            break
        except RuntimeError as e:
            if 'CUDA out of memory' in f'{e}':
                if batch_size > 1:
                    print(
                        f'CUDA out of memory. Try to decrease batch size from {batch_size} to {batch_size // 2}')
                    batch_size //= 2
                    gc.collect()
                else:
                    print('You don\'t have enough gpu memory to train this model')
                    exit(1)
            else:
                print('Runtime Error', e.args)
                exit(1)
    model.save_pretrained('./output_dir')


def main():
    args = get_args()
    # Random Seed를 설정 해준다. => 재현을 위해
    set_seed(args.seed)
    # FireBase 초기화
    init_firebase()
    # 초기 모델을 다운로드 받는다.
    download_model(args)
    # 데이터를 다운로드 받는다. ( 리뷰 필요 )
    args.train_df = download_data()
    # 학습을 시작 한다.
    train_single(args)
    # 학습 완료된 모델을 업로드 한다.
    upload_model(args)


if __name__ == '__main__':
    main()
