import logging
import os

import requests
import kaggle


def download_kaggle_dataset(dataset: str, data_dir: str = '.data/'):
    sub_folder = '/dataset'
    full_dir = os.getenv('GLOSSA_DATA', data_dir) + sub_folder

    if not os.path.exists(full_dir):
        os.mkdir(full_dir)

    logging.info(f'Fetching Dataset {dataset} from Kaggle.')
    kaggle.api.dataset_download_files(dataset, full_dir, unzip=True)
    return


def download_github_pretrained(url: str, filename: str, data_dir: str = '.data/'):
    sub_folder = '/pretrained'
    full_dir = os.getenv('GLOSSA_DATA', data_dir) + sub_folder

    if not os.path.exists(full_dir):
        os.mkdir(full_dir)

    if not os.path.exists(full_dir + filename):
        logging.info(f'Fetching Pretrained {filename} from Github.')
        response = requests.get(url)
        if response.status_code == 200:
            with open(full_dir + filename, 'wb') as file:
                file.write(response.content)
        else:
            return ''
    return full_dir + filename



