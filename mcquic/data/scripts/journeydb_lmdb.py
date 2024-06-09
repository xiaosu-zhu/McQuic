import jsonlines
import sys
import os
import pickle
import lmdb
import tarfile
from io import BytesIO
import json
from joblib import Parallel, delayed
from PIL import Image
import PIL
import datasets
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from tqdm import tqdm


def main(path):
    env = lmdb.Environment(os.path.join(path, 'data', 'train', 'lmdb'), subdir=True, map_size=int(1024 ** 4))
    i = 0
    with env.begin(write=True) as txn:
        with jsonlines.open(os.path.join(path, 'data', 'train', 'train_anno_realease_repath.jsonl')) as reader:
            for obj in reader:
                if i % 1000 == 0:
                    print(i)
                i += 1
                txn.put(obj['img_path'].removeprefix('./').removesuffix('.jpg').encode('utf-8'), pickle.dumps(obj))
    env.close()


def rewrite_tars(src, tgt, chunk):
    os.makedirs(os.path.join(tgt, 'imgs'), exist_ok=True)
    srcTar = tarfile.open(os.path.join(src, 'imgs', f"{chunk:03d}.tgz"), 'r:gz')
    tgtTar = tarfile.open(os.path.join(tgt, 'imgs', f"{chunk:03d}.tgz"), 'w:gz')

    env = lmdb.Environment(os.path.join(src, 'lmdb'), subdir=True, map_size=int(1024 ** 4), readonly=True, readahead=False, meminit=False, max_spare_txns=1, lock=False)

    with env.begin(write=False, buffers=True) as txn:
        for srcInfo in srcTar:
            if srcInfo.type != tarfile.REGTYPE:
                # tgtTar.addfile(srcInfo)
                continue

            if not srcInfo.name.endswith('jpg'):
                continue

            # 000/[UUID].jpg
            name = srcInfo.name
            key = name.removesuffix('.jpg')
            record = pickle.loads(txn.get(key.encode('utf-8')))

            if ('Task2' not in record) or ('Caption' not in record['Task2']) or (record['Task2']['Caption'] is None) or (len(record['Task2']['Caption']) < 5):
                continue



            # write file content
            file_content = srcTar.extractfile(srcInfo).read()

            file_io = BytesIO(file_content)
            try:
                a = Image.open(file_io)
            except:
                continue


            new_tarinfo = tarfile.TarInfo(name=srcInfo.name.replace('/', '_'))
            new_tarinfo.size = len(file_content)
            tgtTar.addfile(new_tarinfo, BytesIO(file_content))



            # json_str = json.dumps(record['Task2']['Caption'])
            json_file_like_obj = BytesIO(record['Task2']['Caption'].encode('utf-8'))

            record_tar_info = tarfile.TarInfo(name=f"{key.replace('/', '_')}.txt")
            record_tar_info.size = len(record['Task2']['Caption'].encode('utf-8'))
            tgtTar.addfile(record_tar_info, json_file_like_obj)
    env.close()
    srcTar.close()
    tgtTar.close()


from mcquic.data.transforms import (
    getTrainingPreprocess,
    getEvalTransform,
    getTrainingPreprocessWithText,
)
def wdsJouneyDBWithLabel(sample):
    text = sample['txt']
    # caption = f"a photo of {label}"
    image = sample["jpg"].convert("RGB")

    return {"jpeg": image, "label": text}


def test_dataset(tgt):
    dataset = datasets.load_dataset('webdataset', data_dir=tgt, split="train", streaming=True).map(wdsJouneyDBWithLabel).map(getTrainingPreprocessWithText()).remove_columns(["jpg"])
    loader = DataLoader(dataset, batch_size=48, num_workers=24)
    for sample in tqdm(loader):
        continue
    print('test passed')

if __name__ == '__main__':
    path = sys.argv[1]
    if len(sys.argv) > 2:
        tgt = sys.argv[2]
        result = Parallel(32)(delayed(rewrite_tars)(os.path.join(path, 'data', 'train'), os.path.join(tgt, 'data', 'train'), i) for i in range(200))
        test_dataset(tgt)
    else:
        main(path)
