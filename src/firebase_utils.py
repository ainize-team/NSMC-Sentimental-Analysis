import json
import os

import firebase_admin
import pandas as pd
from firebase_admin import credentials, db, storage

MODEL_FILE_LIST = [
    'tokenizer.json',
    'special_tokens_map.json',
    'pytorch_model.bin',
    'config.json',
]


def init_firebase():
    cred = credentials.Certificate("keys/mlops-crawler-firebase.json")
    with open('keys/firebase-config.json') as f:
        data = json.load(f)
        firebase_admin.initialize_app(cred, data)
    print('Initialize Firebase')


def download_model(model_name, model_version):
    bucket = storage.bucket()
    if not os.path.exists('./model'):
        os.makedirs('./model')
    for file_name in MODEL_FILE_LIST:
        bucket.blob(f'model/{model_name}/{model_version}/{file_name}'). \
            download_to_filename(f'./model/{file_name}')
    print(f'Download {model_name} Version {model_version}')


def download_data():
    ref = db.reference()
    data = ref.get()
    document_list = []
    label_list = []
    # 전처리된 데이터가 있으며, text가 있는 경우
    for key, value in data['raw'].items():
        if key in data['preprocessed'] and len(data['preprocessed']) > 0:
            if value['score'] > 8:
                document_list.append(data['preprocessed'][key])
                label_list.append(1)
            elif value['score'] < 5:
                document_list.append(data['preprocessed'][key])
                label_list.append(0)

    train_df = pd.DataFrame()
    train_df['document'] = document_list
    train_df['label'] = label_list
    print(f'Data Load : {len(document_list)}')
    print(f'Positive : {sum(label_list)}, Negative : {len(document_list) - sum(label_list)}')

    val_document_list = []
    val_label_list = []
    for key, value in data['validationData'].items():
        val_document_list.append(value['text'])
        val_label_list.append(value['label'])

    val_df = pd.DataFrame()
    val_df['document'] = val_document_list
    val_df['label'] = val_label_list
    return train_df, val_df


def upload_model(model_name, output_version):
    bucket = storage.bucket()
    for file_name in MODEL_FILE_LIST:
        bucket.blob(f'model/{model_name}/{output_version}/{file_name}'). \
            upload_from_filename(f'./output/{file_name}')
    print(f'Upload Model at {model_name} Version {output_version}')


def get_train_tasks():
    return db.reference('trainTask').get()


def delete_train_tasks(key):
    db.reference(f'trainTask/{key}').delete()


def upload_result(value, val_loss, val_acc):
    bucket = storage.bucket()
    with open('result.json','w') as f:
        json.dump({
            'maxSeqLen': value.get('maxSeqLen', 128),
            'batchSize': value.get('batchSize', 32),
            'numTrainEpochs': value.get('numTrainEpochs', 10),
            'learningRate': value.get('learningRate', 5e-5),
            'warmupProportion': value.get('warmupProportion', 0.0),
            'validationLoss': val_loss,
            'validationAccuracy': val_acc,
        }, f, indent=2, ensure_ascii=False)
    bucket.blob(f'model/{value["modelName"]}/{value["outputVersion"]}/result.json'). \
        upload_from_filename('result.json')
    print(f'Upload Result at {value["modelName"]} Version {value["outputVersion"]}')
