import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import argparse
import json
import multiprocessing as mp
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.data import read_problems, write_jsonl
from utils.em_inf_utils import AdaptiveTemperatureProcessor, min_entropy_inference
from utils.evaluation_leetcode import evaluate_functional_correctness
from vllm import LLM, SamplingParams

os.environ["TOKENIZERS_PARALLELISM"] = "false"
version = "20240121-Jul"
STOP_WORDS = ["\nassert", "assert"]
DATA_DIR = Path(__file__).parent / "data"


def match_code(s):
    pattern = r"```python(.*?)```"
    sol = re.findall(pattern, s, re.DOTALL)
    if len(sol) > 0:
        for code_block in reversed(sol):
            if "def " in code_block:
                return code_block
        return sol[-1]

    pattern = r"```(.*?)```"
    sol = re.findall(pattern, s, re.DOTALL)
    if len(sol) > 0:
        for code_block in reversed(sol):
            if "def " in code_block:
                return code_block
        return sol[-1]

    return s.split("```")[0]


def generate_sample_batch(question_list):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.7,
        # max_model_len=10000,
        max_model_len=4096,
    )
    sampling_params = SamplingParams(
        max_tokens=8192,
        temperature=args.temperature,
        seed=42,
        n=1,
        stop=["<|eot_id|>"],
    )

    outputs = llm.generate(question_list, sampling_params, use_tqdm=True)
    completions = [
        "```python\n" + match_code(output.outputs[0].text) + "\n```"
        for output in outputs
    ]
    outputs = [output.outputs[0].text for output in outputs]
    return completions, outputs


def code_preprocessing(outputs):
    completions = ["```python\n" + match_code(output) + "\n```" for output in outputs]
    return completions, outputs


def make_conv_hf(example, tokenizer):
    system_prompt = open("system_prompt.md").read()
    system_prompt = "\nWhen tackling complex reasoning tasks, you have access to the following actions. Use them as needed to progress through your thought process.\n\n[ASSESS]\n\n[ADVANCE]\n\n[VERIFY]\n\n[SIMPLIFY]\n\n[SYNTHESIZE]\n\n[PIVOT]\n\n[OUTPUT]\n\nYou should strictly follow the format below:\n\n[ACTION NAME]\n\n# Your action step 1\n\n# Your action step 2\n\n# Your action step 3\n\n...\n\nNext action: [NEXT ACTION NAME]\n"
    msg = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": example["prompt_sft"]
            + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end.",
        },
    ]
    chat = tokenizer.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True
    )
    return chat


def extract_python_code(generation: str):
    generation = generation.replace("[PYTHON]", "```python").replace("[/PYTHON]", "```")
    if "```python" in generation:
        p_code = re.compile(r"```python\n(.*?)\n```", flags=re.DOTALL)
        code_block = p_code.findall(generation)[0]
        return code_block
    else:
        codelist = re.split("\ndef|\nclass|\nif|\n#|\nprint", generation)
        return codelist[0]


def evaluate_main(
    generation_path: str,
    result_path: str,
    temp_dir: str,
    total_generated_tokens: int = None,
):
    problem_path = args.input_data
    problems = [json.loads(line) for line in open(problem_path, "r", encoding="utf-8")]

    id2problems = {x["task_id"]: x for x in problems}

    results = [json.loads(line) for line in open(generation_path, "r")]
    for result in results:
        if "task_id" not in result:
            result["task_id"] = problems[result["index"]]["task_id"]

        if "generation" not in result:
            try:
                if "output" not in result:
                    result["output"] = result["response"]
                if result["output"].startswith("\n        "):
                    func_code = extract_python_code(result["prompt_sft"]).strip()
                    result["generation"] = func_code + "\n" + result["output"]
                else:
                    result["generation"] = extract_python_code(result["output"])
            except:
                result["generation"] = result["output"]

    with open(result_path, "w") as fr:
        for result in results:
            fr.write(json.dumps(result) + "\n")

    score = evaluate_functional_correctness(
        input_file=result_path,
        tmp_dir=temp_dir,
        problem_file=problem_path,
        result_path=result_path,
    )

    hardness_results = defaultdict(int)
    for result in [json.loads(line) for line in open(result_path, "r")]:
        problem = id2problems[result["task_id"]]

        hardness = problem["meta"]["difficulty"]
        hardness_results[hardness] += 1
        hardness_results[hardness + "_correct"] += result["passed"]

    print("=" * 100)
    print("Evaluate {} over.".format(generation_path))
    print("Pass@1: {:.3f}".format(score["pass@1"]))
    for key in ["Easy", "Medium", "Hard"]:
        if key.endswith("_correct"):
            continue
        acc = hardness_results[key + "_correct"] / hardness_results[key]
        print(
            "{}: {:.3f}({}/{})".format(
                key, acc, hardness_results[key + "_correct"], hardness_results[key]
            )
        )

    if args.inference_mode == "em_inf":
        threshold_str = str(hyperparameters["threshold"])
        kl_weight_str = str(hyperparameters["kl_weight"])
        grad_steps_str = str(hyperparameters["n_grad_steps"])
        lr_str = str(hyperparameters["learning_rate"])
        file_name = (
            "result_em_inf"
            + f"_threshold_{threshold_str}_klweight_{kl_weight_str}_gradsteps_{grad_steps_str}_lr_{lr_str}"
            + ".txt"
        )
        score_path = os.path.join(args.save_dir, file_name)
    elif args.inference_mode == "adaptive_temp":
        ratio_str = str(hyperparameters["target_ratio"])
        threshold_str = str(hyperparameters["threshold"])
        file_name = (
            "results_adaptive_temp"
            + f"_threshold_{threshold_str}_ratio_{ratio_str}"
            + ".txt"
        )
        score_path = os.path.join(args.save_dir, file_name)
    else:
        score_path = os.path.join(args.save_dir, "result.txt")

    with open(score_path, "w") as f:
        f.write("Pass@1: {:.3f}\n".format(score["pass@1"]))
        for key in ["Easy", "Medium", "Hard"]:
            if key.endswith("_correct"):
                continue
            acc = hardness_results[key + "_correct"] / hardness_results[key]
            f.write(
                "{}: {:.3f}({}/{})\n".format(
                    key, acc, hardness_results[key + "_correct"], hardness_results[key]
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--num-samples-per-task", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--inference_mode", "-i", type=str, default="normal")
    parser.add_argument(
        "--hyperparameters",
        type=str,
        default='{"threshold": 0.1, "kl_weight": 0.001, "learning_rate": 0.1, "n_grad_steps": 3}',
    )
    args = parser.parse_args()

    problems = pd.read_json(args.input_data, lines=True)

    samples = []
    del problems["start_time"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    problems["instruction"] = problems.apply(
        lambda row: make_conv_hf(row, tokenizer), axis=1
    )

    if args.inference_mode == "normal":
        completions, outputs = generate_sample_batch(problems["instruction"])
    elif args.inference_mode == "self_refinement":
        root_dir = os.path.dirname(args.save_dir)
        vllm_model_instance = LLM(
            model=args.model,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype="bfloat16",
            enforce_eager=True,
            skip_tokenizer_init=False,
            gpu_memory_utilization=0.7,
            # max_model_len=10000
        )
        for cur_iter in range(4):
            print("*" * 30)
            print(
                f"Generating with self-refinement inference mode: iteration {cur_iter}"
            )

            problems = pd.read_json(args.input_data, lines=True)

            samples = []
            del problems["start_time"]

            tokenizer = AutoTokenizer.from_pretrained(args.model)
            problems["instruction"] = problems.apply(
                lambda row: make_conv_hf(row, tokenizer), axis=1
            )

            # get previous iteration's prompt and completions if current_iter > 0:
            question_list = problems["instruction"]
            current_prompt_prefix = [""] * len(question_list)
            if cur_iter > 0:
                print("Loading previous iteration's prompts...")
                previous_iter_dir = os.path.join(
                    root_dir, f"iter_{cur_iter - 1}", "leetcode_chat"
                )

                # read previous iteration's concatenated prompts
                current_prompt_prefix = []
                with open(
                    os.path.join(previous_iter_dir, "self_refinement_prompts.jsonl"),
                    "r",
                    encoding="utf-8",
                ) as f:
                    for line in f:
                        data = json.loads(line)
                        current_prompt_prefix.append(data["concatenated_prompt"])

            current_question_list = []
            for prefix, question in zip(current_prompt_prefix, question_list):
                current_question_list.append(prefix + question)
            print(f"example prompt: {current_question_list[0]}")

            # completion
            sampling_params = SamplingParams(
                max_tokens=8192,
                temperature=args.temperature,
                seed=42,
                stop=["\n###\nProblem: ", "<|eot_id|>"],
            )
            outputs = vllm_model_instance.generate(
                current_question_list, sampling_params, use_tqdm=True
            )

            total_generated_tokens = 0
            for output in outputs:
                tokens = tokenizer(output.outputs[0].text)["input_ids"]
                total_generated_tokens += len(tokens)

            completions = [
                "```python\n" + match_code(output.outputs[0].text) + "\n```"
                for output in outputs
            ]
            outputs = [output.outputs[0].text for output in outputs]

            # set current iteration's saving path
            args.save_dir = os.path.join(root_dir, f"iter_{cur_iter}", "leetcode_chat")
            os.makedirs(args.save_dir, exist_ok=True)
            print(f"current iteration's saving path: {args.save_dir}")

            # save current iteration's prompts and completions together into a jsonl file, where each line is a dict: {"id": id, "prompt": prompt}
            tmp_data = []
            for current_prompt, model_output in zip(current_question_list, completions):
                tmp_data.append(
                    {
                        "concatenated_prompt": current_prompt
                        + model_output
                        + tokenizer.eos_token
                        + "\n\n",
                    }
                )
            with open(
                os.path.join(args.save_dir, "self_refinement_prompts.jsonl"),
                "w",
                encoding="utf-8",
            ) as f:
                for line in tmp_data:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")

            for i in range(len(completions)):
                if "class Solution:" not in completions[i]:
                    completions[i] = (
                        completions[i].replace("```python", "").replace("```", "")
                    )
                    completions[i] = "\n".join(
                        ["    " + line for line in completions[i].split("\n")]
                    )
                    completions[i] = "class Solution:\n" + completions[i]
                    completions[i] = "```python\n" + completions[i] + "\n```"

            problems["output"] = completions
            problems["raw_output"] = outputs
            samples = problems.to_dict(orient="records")
            output_filepath = os.path.join(args.save_dir, "samples.jsonl")
            write_jsonl(output_filepath, samples)

            result_path = output_filepath.replace(".jsonl", "_result.jsonl")
            try:
                # Set the start method to 'fork'
                # Use force=True if it might have been set to 'spawn' previously in the same run
                mp.set_start_method("fork", force=True)
                print("Multiprocessing start method set to 'fork'.")
            except (RuntimeError, ValueError) as e:
                # Handle cases where it might fail (e.g., already started processes)
                print(f"Could not set start method to 'fork': {e}")
                # Optionally check the current method
                current_method = mp.get_start_method()
                print(f"Using current start method: {current_method}")
            evaluate_main(
                output_filepath,
                result_path,
                temp_dir="output/temp",
                total_generated_tokens=total_generated_tokens,
            )

        exit(0)

    elif args.inference_mode == "ice_self_refinement":
        root_dir = os.path.dirname(args.save_dir)
        for cur_iter in range(4):
            print("*" * 30)
            print(
                f"Generating with self-refinement inference mode: iteration {cur_iter}"
            )

            problems = pd.read_json(args.input_data, lines=True)

            samples = []
            del problems["start_time"]

            tokenizer = AutoTokenizer.from_pretrained(args.model)
            problems["instruction"] = problems.apply(
                lambda row: make_conv_hf(row, tokenizer), axis=1
            )

            # get previous iteration's prompt and completions if current_iter > 0:
            question_list = problems["instruction"]
            current_prompt_prefix = [""] * len(question_list)
            if cur_iter > 0:
                print("Loading previous iteration's prompts...")
                previous_iter_dir = os.path.join(
                    root_dir, f"iter_{cur_iter - 1}", "leetcode_chat"
                )

                # read previous iteration's concatenated prompts
                current_prompt_prefix = []
                with open(
                    os.path.join(previous_iter_dir, "self_refinement_prompts.jsonl"),
                    "r",
                    encoding="utf-8",
                ) as f:
                    for line in f:
                        data = json.loads(line)
                        current_prompt_prefix.append(data["concatenated_prompt"])

            current_question_list = []
            for prefix, question in zip(current_prompt_prefix, question_list):
                current_question_list.append(prefix + question)
            print(f"example prompt: {current_question_list[0]}")

            # completion
            ################################# Data Parallelism Inference setup #################################
            try:
                mp.set_start_method("spawn", force=True)  # Important for CUDA safety!
            except:
                print("Start method 'spawn' already set.")
            num_processes = 14
            ####################################################################################################
            # --- 1. Prepare and Chunkify prompts ---
            k, m = divmod(len(current_question_list), num_processes)
            prompt_chunks = [
                current_question_list[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
                for i in range(num_processes)
            ]

            # --- 2. Create and Start Processes ---
            print(
                f"\nStarting inference on {len(current_question_list)} prompts across {num_processes} processes..."
            )
            hyperparameters = {
                "threshold": 0.1,
                "kl_weight": 0.001,
                "learning_rate": 0.1,
                "n_grad_steps": 3,
            }
            with mp.Pool(processes=num_processes) as pool:
                all_completions = pool.starmap(
                    min_entropy_inference,
                    [
                        (
                            args.model,
                            chunk,
                            i,
                            hyperparameters,
                            {"task": "leetcode", "mode": "ice_min_entropy"},
                            8192,
                        )
                        for i, chunk in enumerate(prompt_chunks)
                    ],
                )

            outputs = [output for sublist in all_completions for output in sublist]

            total_generated_tokens = 0
            for output in outputs:
                tokens = tokenizer(output)["input_ids"]
                total_generated_tokens += len(tokens)

            completions, outputs = code_preprocessing(outputs)

            # set current iteration's saving path
            args.save_dir = os.path.join(root_dir, f"iter_{cur_iter}", "leetcode_chat")
            os.makedirs(args.save_dir, exist_ok=True)
            print(f"current iteration's saving path: {args.save_dir}")

            # save current iteration's prompts and completions together into a jsonl file, where each line is a dict: {"id": id, "prompt": prompt}
            tmp_data = []
            for current_prompt, model_output in zip(current_question_list, completions):
                tmp_data.append(
                    {
                        "concatenated_prompt": current_prompt
                        + model_output
                        + tokenizer.eos_token
                        + "\n\n",
                    }
                )
            with open(
                os.path.join(args.save_dir, "self_refinement_prompts.jsonl"),
                "w",
                encoding="utf-8",
            ) as f:
                for line in tmp_data:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")

            for i in range(len(completions)):
                if "class Solution:" not in completions[i]:
                    completions[i] = (
                        completions[i].replace("```python", "").replace("```", "")
                    )
                    completions[i] = "\n".join(
                        ["    " + line for line in completions[i].split("\n")]
                    )
                    completions[i] = "class Solution:\n" + completions[i]
                    completions[i] = "```python\n" + completions[i] + "\n```"

            problems["output"] = completions
            problems["raw_output"] = outputs
            samples = problems.to_dict(orient="records")
            output_filepath = os.path.join(args.save_dir, "samples.jsonl")
            write_jsonl(output_filepath, samples)

            result_path = output_filepath.replace(".jsonl", "_result.jsonl")
            try:
                # Set the start method to 'fork'
                # Use force=True if it might have been set to 'spawn' previously in the same run
                mp.set_start_method("fork", force=True)
                print("Multiprocessing start method set to 'fork'.")
            except (RuntimeError, ValueError) as e:
                # Handle cases where it might fail (e.g., already started processes)
                print(f"Could not set start method to 'fork': {e}")
                # Optionally check the current method
                current_method = mp.get_start_method()
                print(f"Using current start method: {current_method}")
            evaluate_main(
                output_filepath,
                result_path,
                temp_dir="output/temp",
                total_generated_tokens=total_generated_tokens,
            )

        exit(0)

    elif args.inference_mode == "em_inf":
        ################################# Data Parallelism Inference setup #################################
        mp.set_start_method("spawn")  # Important for CUDA safety!
        num_processes = 16

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        ####################################################################################################

        # --- 1. Prepare and Chunkify prompts ---
        prompts = problems["instruction"]
        k, m = divmod(len(prompts), num_processes)
        prompt_chunks = [
            prompts[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
            for i in range(num_processes)
        ]

        # --- 2. Create and Start Processes ---
        hyperparameters = json.loads(args.hyperparameters)
        print(
            f"Running with EM-INF inference mode using hyperparameters {hyperparameters}"
        )

        with mp.Pool(processes=num_processes) as pool:
            results_list = pool.starmap(
                min_entropy_inference,
                [
                    (args.model, chunk, i, args.temperature, hyperparameters, 8192)
                    for i, chunk in enumerate(prompt_chunks)
                ],
            )

        outputs = [output for sublist in results_list for output in sublist]
        completions, outputs = code_preprocessing(outputs)

    elif args.inference_mode == "adaptive_temp":
        hyperparameters = json.loads(args.hyperparameters)
        print(
            f"Running with adaptive temperature inference mode using hyperparameters {hyperparameters}"
        )

        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            tensor_parallel_size=4,
            dtype="bfloat16",
            gpu_memory_utilization=0.7,
        )
        logits_processor = AdaptiveTemperatureProcessor(
            tmax=hyperparameters["tmax"],
            tmin=hyperparameters["tmin"],
            max_iter=hyperparameters["max_iter"],
            tol=hyperparameters["tol"],
            target_ratio=hyperparameters["target_ratio"],
            target_threshold=hyperparameters["threshold"],
        )
        sampling_params = SamplingParams(
            max_tokens=4096,
            temperature=1.0,
            seed=42,
            logits_processors=[logits_processor],
            stop=["\n###\nProblem: ", "<|eot_id|>"],
        )

        prompts = problems["instruction"]
        vllm_outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        outputs = [output.outputs[0].text for output in vllm_outputs]
        completions, outputs = code_preprocessing(outputs)

    else:
        print(f"Invalid mode: {args.inference_mode}")

    for i in range(len(completions)):
        if "class Solution:" not in completions[i]:
            completions[i] = completions[i].replace("```python", "").replace("```", "")
            completions[i] = "\n".join(
                ["    " + line for line in completions[i].split("\n")]
            )
            completions[i] = "class Solution:\n" + completions[i]
            completions[i] = "```python\n" + completions[i] + "\n```"

    problems["output"] = completions
    problems["raw_output"] = outputs
    samples = problems.to_dict(orient="records")
    output_filepath = os.path.join(args.save_dir, "samples.jsonl")
    write_jsonl(output_filepath, samples)

    result_path = output_filepath.replace(".jsonl", "_result.jsonl")
    try:
        # Set the start method to 'fork'
        # Use force=True if it might have been set to 'spawn' previously in the same run
        mp.set_start_method("fork", force=True)
        print("Multiprocessing start method set to 'fork'.")
    except (RuntimeError, ValueError) as e:
        # Handle cases where it might fail (e.g., already started processes)
        print(f"Could not set start method to 'fork': {e}")
        # Optionally check the current method
        current_method = mp.get_start_method()
        print(f"Using current start method: {current_method}")
    evaluate_main(
        output_filepath, result_path, temp_dir=os.path.join(args.save_dir, "temp")
    )
