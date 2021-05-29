import os
import json
import firebase_admin
from firebase_admin import credentials, db, storage
import pandas as pd

MODEL_FILE_LIST = [
    'tokenizer.json',
    'special_tokens_map.json',
    'pytorch_model.bin',
    'config.json',
]


def download_model(args):

    bucket = storage.bucket()
    if not os.path.exists('./model'):
        os.makedirs('./model')
    for file_name in MODEL_FILE_LIST:
        bucket.blob(f'model/{args.model_name}/{args.model_version}/{file_name}'). \
            download_to_filename(f'./model/{file_name}')


def download_data():
    ref = db.reference()
    data = ref.get()
    document_list = []
    label_list = []
    for key, value in data['raw']:
        if value['score'] > 8:
            document_list.append(data['preprocessed'][key])
            label_list.append(1)
        elif value['score'] < 5:
            document_list.append(data['preprocessed'][key])
            label_list.append(0)
    df = pd.DataFrame()
    df['document'] = document_list
    df['label'] = label_list
    return df


def upload_model(args):
    bucket = storage.bucket()
    for file_name in MODEL_FILE_LIST:
        bucket.blob(f'model/{args.model_name}/{args.output_version}/{file_name}').\
            upload_from_filename(f'./output/{file_name}')