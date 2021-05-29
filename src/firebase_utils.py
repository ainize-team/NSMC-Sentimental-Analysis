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


def download_model(args):
    bucket = storage.bucket()
    if not os.path.exists('./model'):
        os.makedirs('./model')
    for file_name in MODEL_FILE_LIST:
        bucket.blob(f'model/{args.model_name}/{args.model_version}/{file_name}'). \
            download_to_filename(f'./model/{file_name}')
    print(f'Download {args.model_name} Version {args.model_version}')


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
    df = pd.DataFrame()
    df['document'] = document_list
    df['label'] = label_list
    print(f'Data Load : {len(document_list)}')
    print(f'Positive : {sum(label_list)}, Negative : {len(document_list) - sum(label_list)}')
    return df


def upload_model(args):
    bucket = storage.bucket()
    for file_name in MODEL_FILE_LIST:
        bucket.blob(f'model/{args.model_name}/{args.output_version}/{file_name}'). \
            upload_from_filename(f'./output/{file_name}')
    print(f'Upload Model at {args.model_name} Version {args.output_version}')
