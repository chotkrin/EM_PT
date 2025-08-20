<h1 style="text-align: center;">EM-RL Training</h1>

## Installation
Installation instructions for training is the same as veRL. Please follow their install 

## Data Processing
The folder examples/data_preprocess consists of data pre-processing scripts. Each script produces a train.parquet and test.parquet file which is used by the trainer. 

## Training 
Launch commands for EM-RL-Token and EM-RL-Sequence training scripts are present in the folder examples/EM_RL_Trainers. Each script uses wandb and saves the checkpoint in a folder called checkpoints/

