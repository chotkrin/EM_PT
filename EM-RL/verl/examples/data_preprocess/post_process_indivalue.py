import argparse
import json
import os
import random
import re
from collections import defaultdict

import datasets

# from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--local_dir', default='/work/nvme/bcaq/shivama2/data_verl/purl_numina_data')
    # parser.add_argument('--percentage', default=0.1, type=float)
    # parser.add_argument('--hdfs_dir', default=None)

    # args = parser.parse_args()

    data_source = "indievalue/indievalue_refined-stmt-52_polar-q-78_paired_train_v3.jsonl"  # 'PRIME-RL/Eurus-2-RL-Data'
    with open(data_source, "r") as f:
        data = [json.loads(x) for x in f.readlines()]

    # data = []
    # with open(data_source, "r", encoding="utf-8") as file:
    #     for line in file:
    #         data.append(json.loads(line))

    # print(data[0])
    # exit(0)
    # print(len)
    train_dat_new = []
    # test_dataset = []

    for data_ent in data:
        prompt = [{"content": f"{data_ent['prompt']}", "role": "user"}]
        dat_point = {
            "prompt": prompt,
            "data_source": "indivalue",
            "ability": "indivalue",
            "reward_model": {
                "style": "rule",
                "ground_truth": data_ent["chosen_response"],
            },
            "extra_info": {"split": "dummy", "index": 0},
        }
        train_dat_new.append(dat_point)

    random.shuffle(train_dat_new)
    train_dat_new = train_dat_new[:5000]
    val_size = int(len(train_dat_new) * 0.1)
    print(f"train dataset size: {len(train_dat_new)}")

    test_dataset = datasets.Dataset.from_list(train_dat_new[:val_size])
    train_dataset = datasets.Dataset.from_list(train_dat_new[val_size:])

    # print(f"train dataset size: {len(train_dataset)}")
    # ?local_dir = args.local_dir
    # hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(
        os.path.join("/work/nvme/bcaq/shivama2/data_verl/indivalue", "train.parquet")
    )
    test_dataset.to_parquet(
        os.path.join("/work/nvme/bcaq/shivama2/data_verl/indivalue", "test.parquet")
    )

    # Sample 15% from each source
