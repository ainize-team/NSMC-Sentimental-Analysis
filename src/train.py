import gc
import time

import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from firebase_utils import init_firebase, download_model, download_data, upload_model, get_train_tasks, \
    delete_train_tasks
from utils import MODEL_FOR_SEQUENCE_CLASSIFICATION, TOKENIZER_CLASSES
from utils import NSMCDataset, get_dataloader, set_seed, accuracy_score


def train(model, train_dataloader, num_train_epochs, learning_rate, warmup_proportion):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.train()
    global_total_step = len(train_dataloader) * num_train_epochs
    global_step = 0
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=global_total_step * warmup_proportion,
                                                num_training_steps=global_total_step)
    with tqdm(total=global_total_step, unit='step') as t:
        total = 0
        total_loss = 0
        for epoch in range(num_train_epochs):
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

                batch_loss = loss.item() * len(b_input_ids)

                total += len(b_input_ids)
                total_loss += batch_loss

                t.set_postfix(loss='{:.6f}'.format(batch_loss),
                              accuracy='{:.2f}'.format(accuracy_score(out_label_ids, preds) * 100))
                t.update(1)
                del b_input_ids
                del outputs
                del loss


def train_single(model_type, train_df, max_seq_len, batch_size, num_train_epochs, learning_rate, warmup_proportion):
    # device 를 할당 한다.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Model 과 Tokenizer를 불러온다.
    model = MODEL_FOR_SEQUENCE_CLASSIFICATION[model_type].from_pretrained('./model')
    tokenizer = TOKENIZER_CLASSES[model_type].from_pretrained('./model')
    # Dataset 을 만든다.
    train_dataset = NSMCDataset(train_df, tokenizer, max_seq_len)
    tokenizer.save_pretrained('./output_dir', legacy_format=False)
    model.to(device)
    batch_size = batch_size
    while batch_size:
        try:
            train_dataloader = get_dataloader(train_dataset, batch_size, shuffle=True)
            # (model, train_dataloader, num_train_epochs, learning_rate, warmup_proportion)
            train(model, train_dataloader, num_train_epochs, learning_rate, warmup_proportion)
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
    init_firebase()
    while True:
        train_tasks = get_train_tasks()
        if train_tasks:
            for key, value in train_tasks.items():
                try:
                    print(f'Train Task {key}')
                    print('Train Args')
                    print(value)
                    # Random Seed를 설정 해준다. => 재현을 위해
                    set_seed(value.get('seed', 42))
                    # 초기 모델을 다운로드 받는다.
                    download_model(value['modelName'], value['modelVersion'])
                    # 데이터를 다운로드 받는다.
                    train_df = download_data()
                    # 학습을 시작 한다.
                    train_single(
                        value['modelType'],
                        train_df,
                        value['maxSeqLen'],
                        value['batchSize'],
                        value['numTrainEpochs'],
                        value['learningRate'],
                        value['warmupProportion']
                    )
                    # 학습 완료된 모델을 업로드 한다.
                    upload_model(value['modelName'], value['outputVersion'])
                except Exception as e:
                    print('Error :', e)
                finally:
                    delete_train_tasks(key)
        else:
            print('There are no train tasks')
            # 60초간 대기
            time.sleep(60)


if __name__ == '__main__':
    main()
