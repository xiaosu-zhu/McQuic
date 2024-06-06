import jsonlines
import sys
import os
import pickle
import lmdb


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

if __name__ == '__main__':
    path = sys.argv[1]
    main(path)
