---
Paper: 2502.00334
license: cc-by-nc-sa-4.0
dataset_info:
configs:
- config_name: AtomicPhysics
  data_files:
  - split: en
    path: AtomicPhysics/en.jsonl
  - split: zh
    path: AtomicPhysics/zh.jsonl
- config_name: ClassicalElectromagnetism
  data_files:
  - split: en
    path: ClassicalElectromagnetism/en.jsonl
  - split: zh
    path: ClassicalElectromagnetism/zh.jsonl
- config_name: ClassicalMechanics
  data_files:
  - split: en
    path: ClassicalMechanics/en.jsonl
  - split: zh
    path: ClassicalMechanics/zh.jsonl
- config_name: Electrodynamics
  data_files:
  - split: en
    path: Electrodynamics/en.jsonl
  - split: zh
    path: Electrodynamics/zh.jsonl
- config_name: GeometricalOptics
  data_files:
  - split: en
    path: GeometricalOptics/en.jsonl
  - split: zh
    path: GeometricalOptics/zh.jsonl
- config_name: QuantumMechanics
  data_files:
  - split: en
    path: QuantumMechanics/en.jsonl
  - split: zh
    path: QuantumMechanics/zh.jsonl
- config_name: Relativity
  data_files:
  - split: en
    path: Relativity/en.jsonl
  - split: zh
    path: Relativity/zh.jsonl
- config_name: SemiconductorPhysics
  data_files:
  - split: en
    path: SemiconductorPhysics/en.jsonl
  - split: zh
    path: SemiconductorPhysics/zh.jsonl
- config_name: Solid-StatePhysics
  data_files:
  - split: en
    path: Solid-StatePhysics/en.jsonl
  - split: zh
    path: Solid-StatePhysics/zh.jsonl
- config_name: StatisticalMechanics
  data_files:
  - split: en
    path: StatisticalMechanics/en.jsonl
  - split: zh
    path: StatisticalMechanics/zh.jsonl
- config_name: TheoreticalMechanics
  data_files:
  - split: en
    path: TheoreticalMechanics/en.jsonl
  - split: zh
    path: TheoreticalMechanics/zh.jsonl
- config_name: Thermodynamics
  data_files:
  - split: en
    path: Thermodynamics/en.jsonl
  - split: zh
    path: Thermodynamics/zh.jsonl
- config_name: WaveOptics
  data_files:
  - split: en
    path: WaveOptics/en.jsonl
  - split: zh
    path: WaveOptics/zh.jsonl
task_categories:
- question-answering
language:
- en
- zh
pretty_name: UGPhysics
size_categories:
- 10K<n<100K
tags:
- text
- physics
---


# UGPhysics: A Comprehensive Benchmark for Undergraduate Physics Reasoning with Large Language Models

**UGPhysics** is a large-scale and comprehensive benchmark tailored for evaluating the physics problem-solving abilities of LLMs across multiple **U**nder**G**raduate-level **Physics** (UGPhysics) disciplines, comprising 5,520 distinct problems
in three main domains, 13 core subjects, and 59 key topic.



# An Example to load the data

```python
from datasets import load_dataset
dataset=load_dataset("UGPhysics/ugphysics", "AtomicPhysics", split="en")

print(dataset[0])
```

More details on loading and using the data are on our [GitHub page](https://github.com/YangLabHKUST/UGPhysics.git).


If you do find our code helpful or use our benchmark dataset, please cite our paper.

```
@article{xu2025ugphysics,
  title={UGPhysics: A Comprehensive Benchmark for Undergraduate Physics Reasoning with Large Language Models},
  author={Xu, Xin and Xu, Qiyun and Xiao, Tong and Chen, Tianhao and Yan, Yuchen and Zhang, Jiaxin and Diao, Shizhe and Yang, Can and Wang, Yang},
  journal={arXiv preprint arXiv:2502.00334},
  year={2025}
}
```