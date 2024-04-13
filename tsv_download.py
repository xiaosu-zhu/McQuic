from img2dataset import download
import shutil
import os

import glob

if __name__ == "__main__":
    allTSV = glob.glob('openimages/*.tsv')
    for t in allTSV:
        with open(t, 'r') as fp:
            content = fp.readlines()
        content[0] = 'url\tsize\tsuffix\n'
        with open(t, 'w') as fp:
            fp.writelines(content)

    output_dir = os.path.abspath("/mnt/nfs/CMG/OpenImagesV7")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    download(
        # NOTE: cant be too large, or most will be failed.
        processes_count=20,
        thread_count=20,
        url_list="openimages/",
        output_folder=output_dir,
        output_format="files",
        input_format="tsv",
        url_col='url',
        encode_format='jpg',
        skip_reencode=True,
        encode_quality=100,
        resize_mode='no',
        # url_col="URL",
        # caption_col="TEXT",
        extract_exif=False,
        enable_wandb=False,
        number_sample_per_shard=1000,
        distributor="multiprocessing"
    )

    # rm -rf bench
