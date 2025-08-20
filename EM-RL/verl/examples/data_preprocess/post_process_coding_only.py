import re
import os
import datasets
import random
import argparse
from collections import defaultdict
from verl.utils.hdfs_io import copy, makedirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/work/nvme/bcaq/shivama2/data_verl/coding_prime_data')
    # parser.add_argument('--percentage', default=0.15, type=float)
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'PRIME-RL/Eurus-2-RL-Data'
    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset['train']
    test_dataset = dataset['validation']
    code_dataset_val = test_dataset.filter(lambda x: x['ability'] == 'code')
    code_data = train_dataset.filter(lambda x: x['ability'] == 'code')
    for data_ent in code_data:
        data_ent['prompt'] = [entry for entry in data_ent['prompt'] if entry.get('role') != 'system']

    for data_ent in code_dataset_val:
        data_ent['prompt'] = [entry for entry in data_ent['prompt'] if entry.get('role') != 'system']

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    code_data.to_parquet(os.path.join(local_dir, 'train.parquet'))
    code_dataset_val.to_parquet(os.path.join(local_dir, 'test.parquet'))

    