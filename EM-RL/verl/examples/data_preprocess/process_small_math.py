import re
import os
import datasets
import random
import argparse
from collections import defaultdict
from verl.utils.hdfs_io import copy, makedirs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data_verl/numina_data')
    parser.add_argument('--percentage', default=0.1, type=float)
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'RLHFlow/numia_prompt_dpo1'# 'PRIME-RL/Eurus-2-RL-Data'
    dataset1 = datasets.load_dataset(data_source)

    train_dataset1 = dataset1['train']
    # test_dataset = dataset['validation']
    # test_dataset = 

    source_list = set()
    for data_ent in train_dataset1:
        src = data_ent['data_source']
        source_list.add(src)
    # source_list = list(source_list)

    data_source = 'RLHFlow/numia_prompt_dpo2'# 'PRIME-RL/Eurus-2-RL-Data'
    dataset2 = datasets.load_dataset(data_source)
    train_dataset2 = dataset2['train']
    for data_ent in train_dataset2:
            src = data_ent['data_source']
            source_list.add(src)

    train_dat_new = []
    test_dataset = []
    for data_ent in dataset1['train']:
        prompt = [{'content': 'Please reason step by step, and put your final answer within \\boxed{{}}.', 'role':'system'},
                  {'content': f"{data_ent['problem']} Let us think step by step and output the final answer within \\boxed{{}}", 'role':'user'}]
        dat_point = {'prompt':prompt,
         'data_source': data_ent['data_source'],
         'ability': 'math',
         'reward_model': {'style': 'rule', 'ground_truth': data_ent['gt']},
         'extra_info': {'split': 'dummy', 'index': 0}}
        train_dat_new.append(dat_point)
    
    for data_ent in dataset2['train']:
        prompt = [{'content': 'Please reason step by step, and put your final answer within \\boxed{{}}.', 'role':'system'},
                  {'content': f"{data_ent['problem']} Let us think step by step and output the final answer within \\boxed{{}}", 'role':'user'}]
        dat_point = {'prompt':prompt,
         'data_source': data_ent['data_source'],
         'ability': 'math',
         'reward_model': {'style': 'rule', 'ground_truth': data_ent['gt']},
         'extra_info': {'split': 'dummy', 'index': 0}}
        train_dat_new.append(dat_point)
    
    
    random.shuffle(train_dat_new)
    val_size = int(len(train_dat_new) * 0.1)


    test_dataset = datasets.Dataset.from_list(train_dat_new[:val_size])
    train_dataset = datasets.Dataset.from_list(train_dat_new[val_size:])

    # print(f"train dataset size: {len(train_dataset)}")
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    
    # Sample 15% from each source
    sampled_dir = os.path.join(local_dir, 'labeled_samples')
    os.makedirs(sampled_dir, exist_ok=True)

    for source_id in set(train_dataset['data_source']):
        # Filter samples for this source
        # print(len())
        source_data = train_dataset.filter(lambda x: x['data_source'] == source_id)
        # print(len(source_data))
        # Shuffle and sample 15%
        print(f"Sampling {args.percentage} from {len(source_data)} samples")
        sample_size = int(args.percentage * len(source_data))
        sampled_data = source_data.shuffle(seed=42).select(range(sample_size))

        # # Save sampled data
        sample_path = os.path.join(sampled_dir, f'source_{source_id}_sample.parquet')
        sampled_data.to_parquet(sample_path)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
