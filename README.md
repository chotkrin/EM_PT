# The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning

## Evaluation

Before evaluation, please install the required packages following the instructions below.

First, create `virtual_envs` directory under the root directory for installing virtual environments.

```bash
mkdir virtual_envs
```

We use different virtual environments for different test sets. The reason for this is that the version of the package is critical when evaluating the math and code benchmarks, and we build virtual environments from the official repositories of [Eurus](https://github.com/OpenBMB/Eurus), [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math), and [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench), respectively.

For AIME 2024, AMC, MATH-500, and LeetCode, we use the following virtual environment.

```bash
python -m venv ./virtual_envs/prime
source ./virtual_envs/prime/bin/activate
pip install -r ./eval/requirements_prime.txt
```

For Minerva Math and OlympiadBench, we use the following virtual environment.

```bash
python -m venv ./virtual_envs/qwen_math
source ./virtual_envs/qwen_math/bin/activate
pip install -r ./eval/requirements_qwen_math.txt
```

For LiveCodeBench, we use the following virtual environment.

```bash
python -m venv ./virtual_envs/lcb
source ./virtual_envs/lcb/bin/activate
pip install -r ./eval/requirements_lcb.txt
```

### Running Evaluations

The directory `evaluation_scripts` contains the shell scripts to run evaluation with different inference mode. Different evaluations mode require different arguments. Please follow the guide below to setup the evaluation properly.

#### 1. Normal Inference mode

`evaluation_scripts/normal_eval.sh` contains the settings and commands for running evaluation with normal inference mode. Please set the following arguments in `evaluation_scripts/normal_eval.sh` accordingly:

- `MODEL_NAME_OR_PATH`: Huggingface model name (e.g. Qwen/Qwen2.5-7B-Instruct) or the path to your local model checkpoint.
- `TEMP`: set the temperature.
- `TASK`: set the tasks you want to evaluate. Available tasks for Normal inference are `math500`, `amc`, `aime`, `qwen`, `leetcode`, `livecodebench`, `ugphysics`, and `scicode`.
  - `all`: run all the tasks.
  - `math`: run all the math benchmarks, including MATH500, AMC, AIME, and Qwen Math (Minerva and OlympiadBench).
  - `code`: run LeetCode and LiveCodeBench.
  - `amc,math500,leetcode`: run selected tasks, separated by comma.

After the arguements are set, run the following bash command, and the results will be saved in `results/sanitized_model_name/normal`:

```bash
bash evaluation_scripts/normal_eval.sh
```

#### 2. EM-INF Inference mode

`evaluation_scripts/em_inf_eval.sh` contains the settings and commands for running evaluation with EM-INF inference mode. Please set the following arguments in `evaluation_scripts/em_inf_eval.sh` accordingly:

- `MODEL_NAME_OR_PATH`: Huggingface model name (e.g. Qwen/Qwen2.5-7B-Instruct) or the path to your local model checkpoint.
- `TEMP`: set the temperature.
- `TASK`: set the tasks you want to evaluate. Available tasks for EM-INF are `math500`, `amc`, `aime`, `qwen`, `leetcode`, `ugphysics`, and `scicode`.
  - `all`: run all the tasks.
  - `math`: run all the math benchmarks, including MATH500, AMC, AIME, and Qwen Math (Minerva and OlympiadBench).
  - `math,leetcode,ugphysics`: run selected tasks, separated by comma.
- `HYPERPARAMETERS`: there are 4 hyperparameters required for EM-INF.
  - `threshold`: float ranging from $[0, 1]$. If entropy of the logits is below `threshold`, it will stop minimizing entropy and start token sampling.
  - `kl_weight`: float ranging from $[0, 1]$. Controls the importance of KL constraint when minimizing entropy of the logits.
  - `learning_rate`: float ranging from $[0, 1]$.
  - `n_grad_steps`: int ranging from $[0, inf)$. Controls the number of optimization steps when minimizing entropy of logits.
- `NUM_PROCESSES`: the number of model copy to run inference in parallel. Please set this number based on your model size and available GPU memory, otherwise it will raise CUDA Out-of-Memory error.

After the arguements are set, run the following bash command, and the results will be saved in `results/sanitized_model_name/em_inf`:

```bash
bash evaluation_scripts/em_inf_eval.sh
```

#### 3. Adaptive Temp Inference mode

`evaluation_scripts/adaptive_temp_eval.sh` contains the settings and commands for running evaluation with Adaptive Temp inference mode. Please set the following arguments in `evaluation_scripts/adaptive_temp_eval.sh` accordingly:

- `MODEL_NAME_OR_PATH`: Huggingface model name (e.g. Qwen/Qwen2.5-7B-Instruct) or the path to your local model checkpoint.
- `TASK`: set the tasks you want to evaluate. Available tasks for Adaptive Temp inference are `math500`, `amc`, `aime`, `qwen`, `leetcode`, `ugphysics`, and `scicode`.
  - `all`: run all the tasks.
  - `math`: run all the math benchmarks, including MATH500, AMC, AIME, and Qwen Math (Minerva and OlympiadBench).
  - `math,leetcode,ugphysics`: run selected tasks, separated by comma.
- `HYPERPARAMETERS`: there are 6 hyperparameters required for Adaptive Temp.
  - `tmax`: float ranging from $[0, 1]$. The temperature upper bound for adaptive temperature optimization, which is also the starting temperature.
  - `tmin`: float ranging from $[0, 1]$. The temperature lower bound for adaptive temperature optimization.
  - `max_iter`: int ranging from $[0, inf)$. The maximum iteration for optimization.
  - `tol`: float ranging from $[0, inf)$. Controls the tolerance of temperature for deciding when to stop adaptive temperature optimizaiton.
  - `target_ratio`: float ranging from $[0, 1]$. Controls the percentage of initial logit entropy you want to reduce to (e.g. `target_ratio=0.85` means the you want to minimize logit entropy to 0.85 of the initial entropy through adaptive temperature).
  - `threshold`: float ranging from $[0, 1]$. If entropy of the logits is below `threshold`, it will stop minimizing entropy and start token sampling.

After the arguements are set, run the following bash command, and the results will be saved in `results/sanitized_model_name/adaptive_temp`:

```bash
bash evaluation_scripts/adaptive_temp_eval.sh
```
