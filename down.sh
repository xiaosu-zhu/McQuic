#!/bin/bash

srun -N 1 -p amd -n 1 --cpus-per-task=32 --nodelist=cpua01 -J split_0 img2dataset --url_list /ssdfs/datahome/tj24011/datasets/raw/coyo-700m/split_0 --input_format "parquet" --url_col "url" --caption_col "text" --output_folder /ssdfs/datahome/tj24011/datasets/raw/coyo-700m-image/split_0 --processes_count 32 --thread_count 64 --skip_reencode=True --resize_mode no --save_additional_columns '["clip_similarity_vitb32","clip_similarity_vitl14","nsfw_score_opennsfw2","nsfw_score_gantman","watermark_score","aesthetic_score_laion_v2"]' --enable_wandb False

srun -N 1 -p amd -n 1 --cpus-per-task=32 --nodelist=cpua02 -J split_1 img2dataset --url_list /ssdfs/datahome/tj24011/datasets/raw/coyo-700m/split_1 --input_format "parquet" --url_col "url" --caption_col "text" --output_folder /ssdfs/datahome/tj24011/datasets/raw/coyo-700m-image/split_1 --processes_count 32 --thread_count 64 --skip_reencode=True --resize_mode no --save_additional_columns '["clip_similarity_vitb32","clip_similarity_vitl14","nsfw_score_opennsfw2","nsfw_score_gantman","watermark_score","aesthetic_score_laion_v2"]' --enable_wandb False

srun -N 1 -p amd -n 1 --cpus-per-task=32 --nodelist=cpua03 -J split_2 img2dataset --url_list /ssdfs/datahome/tj24011/datasets/raw/coyo-700m/split_2 --input_format "parquet" --url_col "url" --caption_col "text" --output_folder /ssdfs/datahome/tj24011/datasets/raw/coyo-700m-image/split_2 --processes_count 32 --thread_count 64 --skip_reencode=True --resize_mode no --save_additional_columns '["clip_similarity_vitb32","clip_similarity_vitl14","nsfw_score_opennsfw2","nsfw_score_gantman","watermark_score","aesthetic_score_laion_v2"]' --enable_wandb False

srun -N 1 -p amd -n 1 --cpus-per-task=32 --nodelist=cpua04 -J split_3 img2dataset --url_list /ssdfs/datahome/tj24011/datasets/raw/coyo-700m/split_3 --input_format "parquet" --url_col "url" --caption_col "text" --output_folder /ssdfs/datahome/tj24011/datasets/raw/coyo-700m-image/split_3 --processes_count 32 --thread_count 64 --skip_reencode=True --resize_mode no --save_additional_columns '["clip_similarity_vitb32","clip_similarity_vitl14","nsfw_score_opennsfw2","nsfw_score_gantman","watermark_score","aesthetic_score_laion_v2"]' --enable_wandb False