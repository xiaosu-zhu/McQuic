from huggingface_hub import HfApi
import os
import sys
from pathlib import Path


def main(file_path):
    api = HfApi(
        endpoint='https://hf-mirror.com',
        token=os.environ['HF_TOKEN'],
    )
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=Path(file_path).name,
        repo_id='vl-group/McQuic-VAE-Encoder',
        repo_type='model'
    )


if __name__ == '__main__':
    file_path = sys.argv[1]
    main(file_path)
