# The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning

## Evaluation

Before running evaluations, please install the required packages following the instructions below.

First, create a `virtual_envs` directory under the root directory for installing virtual environments:

```bash
mkdir virtual_envs
```

We use different virtual environments for different test sets. This is because the package versions are critical when evaluating math and code benchmarks. We build virtual environments from the official repositories of [Eurus](https://github.com/OpenBMB/Eurus), [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math), and [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench), respectively.

For AIME 2024, AMC, MATH-500, and LeetCode, use the following virtual environment:

```bash
python -m venv ./virtual_envs/prime
source ./virtual_envs/prime/bin/activate
pip install -r ./eval/requirements_prime.txt
```

For Minerva Math and OlympiadBench, use the following virtual environment:

```bash
python -m venv ./virtual_envs/qwen_math
source ./virtual_envs/qwen_math/bin/activate
pip install -r ./eval/requirements_qwen_math.txt
```

For LiveCodeBench, use the following virtual environment:

```bash
python -m venv ./virtual_envs/lcb
source ./virtual_envs/lcb/bin/activate
pip install -r ./eval/requirements_lcb.txt
```

For UGPhysics and SciCode, use the EM-PT environment. To run SciCode evaluation, please download [the numeric test results](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR) and save them as `eval/SciCode/eval/data/test_data.h5`.

### Running Evaluations

The `EM-INF` directory contains shell scripts to run evaluations with different inference modes. Different evaluation modes require different arguments. Please follow the guide below to set up the evaluation properly.

#### 1. Normal Inference Mode

`EM-INF/normal_eval.sh` contains the settings and commands for running evaluation with normal inference mode. Please set the following arguments in `EM-INF/normal_eval.sh` accordingly:

- `MODEL_NAME_OR_PATH`: HuggingFace model name (e.g., Qwen/Qwen2.5-7B-Instruct) or the path to your local model checkpoint.
- `TEMP`: Set the temperature.
- `TASK`: Set the tasks you want to evaluate. Available tasks for normal inference are `math500`, `amc`, `aime`, `qwen`, `leetcode`, `livecodebench`, `ugphysics`, and `scicode`.
  - `all`: Run all the available tasks for normal mode.
  - `math`: Run all the math benchmarks, including MATH-500, AMC, AIME, and Qwen Math (Minerva and OlympiadBench).
  - `math,leetcode,ugphysics`: Run selected tasks, separated by commas.

After the arguments are set, run the following bash command. The results will be saved in `results/sanitized_model_name/normal`:

```bash
bash EM-INF/normal_eval.sh
```

#### 2. EM-INF Inference Mode

`EM-INF/em_inf_eval.sh` contains the settings and commands for running evaluation with EM-INF inference mode. Please set the following arguments in `EM-INF/em_inf_eval.sh` accordingly:

- `MODEL_NAME_OR_PATH`: HuggingFace model name (e.g., Qwen/Qwen2.5-7B-Instruct) or the path to your local model checkpoint.
- `TEMP`: Set the temperature.
- `TASK`: Set the tasks you want to evaluate. Available tasks for EM-INF are `math500`, `amc`, `aime`, `qwen`, `leetcode`, `ugphysics`, and `scicode`.
  - `all`: Run all the available tasks for EM-INF mode.
  - `math`: Run all the math benchmarks, including MATH-500, AMC, AIME, and Qwen Math (Minerva and OlympiadBench).
  - `math,leetcode,ugphysics`: Run selected tasks, separated by commas.
- `HYPERPARAMETERS`: There are 4 hyperparameters required for EM-INF:
  - `threshold`: Float ranging from [0, 1]. If the entropy of the logits is below `threshold`, it will stop minimizing entropy and start token sampling.
  - `kl_weight`: Float ranging from [0, 1]. Controls the importance of the KL constraint when minimizing entropy of the logits.
  - `learning_rate`: Float ranging from [0, 1].
  - `n_grad_steps`: Integer ranging from [0, ∞). Controls the number of optimization steps when minimizing entropy of logits.
- `NUM_PROCESSES`: The number of model copies to run inference in parallel. Please set this number based on your model size and available GPU memory, otherwise it will raise a CUDA Out-of-Memory error.

After the arguments are set, run the following bash command. The results will be saved in `results/sanitized_model_name/em_inf`:

```bash
bash EM-INF/em_inf_eval.sh
```

#### 3. Adaptive Temperature Inference Mode

`EM-INF/adaptive_temp_eval.sh` contains the settings and commands for running evaluation with Adaptive Temperature inference mode. Please set the following arguments in `EM-INF/adaptive_temp_eval.sh` accordingly:

- `MODEL_NAME_OR_PATH`: HuggingFace model name (e.g., Qwen/Qwen2.5-7B-Instruct) or the path to your local model checkpoint.
- `TASK`: Set the tasks you want to evaluate. Available tasks for Adaptive Temperature inference are `math500`, `amc`, `aime`, `qwen`, `leetcode`, `ugphysics`, and `scicode`.
  - `all`: Run all the available tasks for Adaptive Temperature mode.
  - `math`: Run all the math benchmarks, including MATH-500, AMC, AIME, and Qwen Math (Minerva and OlympiadBench).
  - `math,leetcode,ugphysics`: Run selected tasks, separated by commas.
- `HYPERPARAMETERS`: There are 6 hyperparameters required for Adaptive Temperature:
  - `tmax`: Float ranging from [0, 1]. The temperature upper bound for adaptive temperature optimization, which is also the starting temperature.
  - `tmin`: Float ranging from [0, 1]. The temperature lower bound for adaptive temperature optimization.
  - `max_iter`: Integer ranging from [0, ∞). The maximum iteration for optimization.
  - `tol`: Float ranging from [0, ∞). Controls the tolerance of temperature for deciding when to stop adaptive temperature optimization.
  - `target_ratio`: Float ranging from [0, 1]. Controls the percentage of initial logit entropy you want to reduce to (e.g., `target_ratio=0.85` means you want to minimize logit entropy to 0.85 of the initial entropy through adaptive temperature).
  - `threshold`: Float ranging from [0, 1]. If the entropy of the logits is below `threshold`, it will stop minimizing entropy and start token sampling.

After the arguments are set, run the following bash command. The results will be saved in `results/sanitized_model_name/adaptive_temp`:

```bash
bash EM-INF/adaptive_temp_eval.sh
```
## EM-RL Training
The folder EM-RL contains the training code for our entropy minimization training method. 

## Acknowledgement: 

Our training code is based on veRL. We use vLLM for inference and deverlop evaluation scripts based on PRIME, Qwen-2.5-Math, LiveCode. Our data is sourced from RLFlow and PRIME. 


