import os

import kaggle


def download_kaggle_dataset(dataset: str, data_dir: str = '.data/'):
    get_dir = os.getenv('GLOSSA_DATA', data_dir)
    if not os.path.exists(get_dir):
        os.mkdir(get_dir)

    kaggle.api.dataset_download_files(dataset, data_dir, unzip=True)
    return
