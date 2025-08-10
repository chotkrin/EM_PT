import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import argparse
import json
import math
import multiprocessing as mp
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.em_inf_utils import AdaptiveTemperatureProcessor, min_entropy_inference
from utils.grader import math_equal
from utils.math_equivalence import is_equiv
from utils.self_consistency_generator import (
    generate_sample_batch_with_n_trajs,
    self_consistency_output_selection,
)
from vllm import LLM, SamplingParams

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def read_jsonl_file(file_path):
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def write_jsonl_file(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def generate_sample_batch(question_list):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="bfloat16",
        gpu_memory_utilization=0.7,
    )
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=args.temp,
        seed=42,
        stop=["\n###\nProblem: ", "<|eot_id|>"],
    )
    outputs = llm.generate(question_list, sampling_params, use_tqdm=True)
    completions = [output.outputs[0].text for output in outputs]
    return completions


def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1 : right_brace_idx].strip()


def match_answer(response):
    is_matched = False

    ans_marker = "the answer is: "
    ans_idx = response.lower().rfind(ans_marker)
    if ans_idx != -1:
        is_matched = True
        response = response[ans_idx + len(ans_marker) :].strip()
        if response.endswith("\n"):
            response = response[:-2]

    ans_marker = "answer:\n"
    ans_idx = response.lower().rfind(ans_marker)
    if ans_idx != -1:
        is_matched = True
        response = response[ans_idx + len(ans_marker) :].strip()
        if response.endswith("\n"):
            response = response[:-2]

    ans_marker = "answer: "
    ans_idx = response.lower().rfind(ans_marker)
    if ans_idx != -1:
        is_matched = True
        response = response[ans_idx + len(ans_marker) :].strip()
        if response.endswith("\n"):
            response = response[:-2]

    # Find boxed
    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed

    # Grade
    return is_matched, response


def make_conv_hf(question, tokenizer):
    system_prompt = open("system_prompt.md").read()
    system_prompt = system_prompt.replace("\n", "")
    content = question
    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    chat = tokenizer.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True
    )
    return chat


def run(args, cur_iter=-1, vllm_model_instance=None):
    outputs = []
    answers = []
    types = []
    levels = []
    matches = []
    fnames_list = []

    cors = {}
    subject_cors = {}
    level_cors = {}
    correct = 0
    total = 0

    all_problems = pd.read_json(
        os.path.join(args.data_dir, "math_test_cleaned.json")
    ).to_dict(orient="records")
    print("reading problems done!")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.inference_mode == "normal":
        print("Using normal inference mode")
        completions = generate_sample_batch(
            [
                make_conv_hf(problem_data["problem"], tokenizer)
                for problem_data in all_problems
            ]
        )

    elif args.inference_mode == "self_consistency":
        print(
            f"Running with self-consistency inference mode using {args.n_trajs} trajectories"
        )
        question_list = [
            make_conv_hf(problem_data["problem"], tokenizer)
            for problem_data in all_problems
        ]
        completions_dict = generate_sample_batch_with_n_trajs(
            args, question_list, args.n_trajs
        )
        completions = self_consistency_output_selection(completions_dict, question_list)

    elif args.inference_mode == "self_refinement":
        print("*" * 30)
        print(f"Running with self-refinement inference mode: iteration {cur_iter}")

        # get previous iteration's prompt and completions if current_iter > 0:
        current_prompt_prefix = [""] * len(all_problems)
        if cur_iter > 0:
            print("Loading previous iteration's prompts...")
            previous_iter_dir = os.path.join(
                args.save_dir, f"iter_{cur_iter - 1}", "math_chat"
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

        question_list = [
            make_conv_hf(problem_data["problem"], tokenizer)
            for problem_data in all_problems
        ]

        current_question_list = []
        for prefix, question in zip(current_prompt_prefix, question_list):
            current_question_list.append(prefix + question)
        print(f"example prompt: {current_question_list[0]}")

        # completion
        sampling_params = SamplingParams(
            max_tokens=4096,
            temperature=args.temp,
            seed=42,
            stop=["\n###\nProblem: ", "<|eot_id|>"],
        )
        vllm_outputs = vllm_model_instance.generate(
            current_question_list, sampling_params, use_tqdm=True
        )
        completions = [output.outputs[0].text for output in vllm_outputs]

        # set current iteration's saving path
        args.save_dir = os.path.join(args.save_dir, f"iter_{cur_iter}", "math_chat")
        os.makedirs(args.save_dir, exist_ok=True)

        # save current iteration's prompts and completions together into a jsonl file, where each line is a dict: {"id": id, "prompt": prompt}
        tmp_data = []
        for problem_data, current_prompt, model_output in zip(
            all_problems, current_question_list, completions
        ):
            tmp_data.append(
                {
                    "original_prompt": problem_data["problem"],
                    "concatenated_prompt": current_prompt + model_output + "\n\n",
                }
            )
        print(
            f"Saving concatenated prompt for next iteration into: {args.save_dir}/self_refinement_prompts.jsonl"
        )
        write_jsonl_file(
            os.path.join(args.save_dir, "self_refinement_prompts.jsonl"), tmp_data
        )

    elif args.inference_mode == "adaptive_temp":
        hyperparameters = json.loads(args.hyperparameters)
        print(
            f"Running with adaptive temperature inference mode using hyperparameters {hyperparameters}"
        )

        conversation_txts = [
            make_conv_hf(problem_data["problem"], tokenizer)
            for problem_data in all_problems
        ]

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

        vllm_outputs = llm.generate(conversation_txts, sampling_params, use_tqdm=True)
        completions = [output.outputs[0].text for output in vllm_outputs]

    elif args.inference_mode == "em_inf":
        ################################# Data Parallelism Inference setup #################################
        mp.set_start_method("spawn")  # Important for CUDA safety!
        num_processes = args.num_processes

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        ####################################################################################################

        # --- 1. Prepare and Chunkify prompts ---
        conversation_txts = [
            make_conv_hf(problem_data["problem"], tokenizer)
            for problem_data in all_problems
        ]
        k, m = divmod(len(conversation_txts), num_processes)
        prompt_chunks = [
            conversation_txts[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
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
                    (args.model, chunk, i, args.temp, hyperparameters)
                    for i, chunk in enumerate(prompt_chunks)
                ],
            )

        # flatten results_list
        completions = [output for sublist in results_list for output in sublist]

        # --- 3. Verification and Cleanup ---
        if len(completions) == len(all_problems):
            print("All completions generated successfully.")
        else:
            print("Missing completions detected.")
    elif args.inference_mode == "ice_self_consistency":
        ################################# Data Parallelism Inference setup #################################
        mp.set_start_method("spawn")  # Important for CUDA safety!
        num_processes = args.num_processes
        ####################################################################################################

        # --- 1. Prepare and Chunkify prompts ---
        conversation_txts = [
            make_conv_hf(problem_data["problem"], tokenizer)
            for problem_data in all_problems
        ]
        upsampled_conversation_txts = conversation_txts * args.n_trajs
        k, m = divmod(len(upsampled_conversation_txts), num_processes)
        prompt_chunks = [
            upsampled_conversation_txts[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
            for i in range(num_processes)
        ]

        # --- 2. Create and Start Processes ---
        print(
            f"\nStarting ICE + self-consistency inference with {args.n_trajs} trajs on {len(conversation_txts)} prompts (total {len(upsampled_conversation_txts)}) across {num_processes} processes..."
        )

        hyperparameters = json.loads(args.min_ent_params)
        print(
            f"Using hyperparameters (is dictionary={isinstance(hyperparameters, dict)}): {hyperparameters}"
        )

        with mp.Pool(processes=num_processes) as pool:
            results_list = pool.starmap(
                min_entropy_inference,
                [
                    (args.model, chunk, i, hyperparameters)
                    for i, chunk in enumerate(prompt_chunks)
                ],
            )

        # Now, "unzip" the list of tuples.
        if results_list:
            # zip(*...) transposes the list of tuples.
            all_completions_tuples, all_prompt_time_dicts_tuples = zip(*results_list)
            all_completions = list(all_completions_tuples)
            all_prompt_time_dicts = list(all_prompt_time_dicts_tuples)

        prompt_time_dict = {}
        for sub_dict in all_prompt_time_dicts:
            for k, v in sub_dict.items():
                if k not in prompt_time_dict:
                    prompt_time_dict[k] = v
                else:
                    prompt_time_dict[k] = prompt_time_dict[k] + v
        prompt_time_dict_path = os.path.join(
            args.save_dir, f"prompt_time_dict_{args.n_trajs}_trajs.json"
        )
        print(f"Saving prompt time dictionary to {prompt_time_dict_path}")
        with open(prompt_time_dict_path, "w", encoding="utf-8") as json_file:
            json.dump(prompt_time_dict, json_file, indent=4)

        outputs = [output for sublist in all_completions for output in sublist]
        print(
            f"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nLength of outputs after merging multiprocesses: {len(outputs)}"
        )

        completions_dict = {}
        total_generated_tokens = 0
        for question, output in zip(upsampled_conversation_txts, outputs):
            if question not in completions_dict:
                completions_dict[question] = []
            completions_dict[question].append(output)

            # tokenize the output_str and count the tokens
            tokens = tokenizer(output)["input_ids"]
            total_generated_tokens += len(tokens)
        print(
            f"Total generated tokens for {args.n_trajs} trajs: {total_generated_tokens}"
        )
        completions, num_consistency = self_consistency_output_selection(
            completions_dict, conversation_txts
        )

    tmp_data = []
    for problem_data, model_output in zip(all_problems, completions):
        problem_data["completion"] = model_output
        tmp_data.append(problem_data)

    if (
        args.inference_mode == "self_consistency"
        or args.inference_mode == "ice_self_consistency"
    ):
        file_name = f"completions_{args.n_trajs}_trajs.jsonl"
    else:
        file_name = "completions.jsonl"
    write_jsonl_file(os.path.join(args.save_dir, file_name), tmp_data)

    for problem_data, model_output in tqdm(zip(all_problems, completions)):
        prob_level = problem_data["level"]
        prob_type = problem_data["type"]
        try:
            prob_level = int(prob_level.split("Level ")[1])
        except:
            prob_level = None

        answer = problem_data["expected_answer"]

        levels.append(prob_level)
        types.append(prob_type)
        is_matched, model_output = match_answer(model_output)
        matches.append(is_matched)
        outputs.append(model_output)
        answers.append(answer)

        try:
            if "\pi" in model_output or "\pi" in answer:
                equivs = []
                for pi in [math.pi, 3.14]:
                    equivs.append(math_equal(model_output, answer, timeout=True, pi=pi))
                equiv = any(equivs)
            else:
                equiv = math_equal(model_output, answer, timeout=True)
        except:
            equiv = False
        fnames_list.append(equiv)
        if (prob_level, prob_type) in cors:
            cors[(prob_level, prob_type)].append(equiv)
        else:
            cors[(prob_level, prob_type)] = [equiv]
        if prob_level in level_cors:
            level_cors[prob_level].append(equiv)
        else:
            if prob_level is not None:
                level_cors[prob_level] = [equiv]
        if prob_type in subject_cors:
            subject_cors[prob_type].append(equiv)
        else:
            if prob_type is not None:
                subject_cors[prob_type] = [equiv]
        if equiv:
            correct += 1

    if (
        args.inference_mode == "self_consistency"
        or args.inference_mode == "ice_self_consistency"
    ):
        file_name = f"results_{args.n_trajs}_trajs.txt"
    elif args.inference_mode == "em_inf":
        threshold_str = str(hyperparameters["threshold"])
        kl_weight_str = str(hyperparameters["kl_weight"])
        grad_steps_str = str(hyperparameters["n_grad_steps"])
        lr_str = str(hyperparameters["learning_rate"])
        file_name = (
            "result_em_inf"
            + f"_threshold_{threshold_str}_klweight_{kl_weight_str}_gradsteps_{grad_steps_str}_lr_{lr_str}"
            + ".txt"
        )
    elif args.inference_mode == "adaptive_temp":
        ratio_str = str(hyperparameters["target_ratio"])
        threshold_str = str(hyperparameters["threshold"])
        file_name = (
            "results_adaptive_temp"
            + f"_threshold_{threshold_str}_ratio_{ratio_str}"
            + ".txt"
        )
    else:
        file_name = "results.txt"
    output_file = os.path.join(args.save_dir, file_name)

    output_dict = {
        "outputs": [],
        "accuracy_by_subject_and_level": defaultdict(list),
        "accuracy_by_level": [],
        "accuracy_by_subject": [],
    }
    print("Match rate: ", np.mean(matches))
    with open(output_file, "w+") as f:
        for k, (output, answer, prob_type, prob_level, match, equiv) in enumerate(
            zip(outputs, answers, types, levels, matches, fnames_list)
        ):
            try:
                f.write(
                    "{} TYPE: {} | LEVEL: {} | OUTPUT: {} | ANSWER: {} | MATCH: {} | CORRECT: {}\n".format(
                        k, prob_type, prob_level, output, answer, match, equiv
                    )
                )
            except:
                f.write("Error line436")
                pass
            output_dict["outputs"].append(
                {
                    "type": prob_type,
                    "level": prob_level,
                    "output": output,
                    "answer": answer,
                    "match": match,
                    "equiv": equiv,
                }
            )

        f.write("#####################\n")
        # also get accuracies for each
        for subject in [
            "Prealgebra",
            "Algebra",
            "Number Theory",
            "Counting & Probability",
            "Geometry",
            "Intermediate Algebra",
            "Precalculus",
        ]:
            for level in range(1, 6):
                key = (level, subject)
                if key not in cors.keys():
                    print("Skipping", key)
                    continue
                cors_list = cors[key]
                print(
                    "{} Level {} Accuracy = {}/{} = {:.3f}".format(
                        subject,
                        level,
                        np.sum(cors_list),
                        len(cors_list),
                        np.mean(cors_list),
                    )
                )
                f.write(
                    "{} Level {} Accuracy = {}/{} = {:.3f}\n".format(
                        subject,
                        level,
                        np.sum(cors_list),
                        len(cors_list),
                        np.mean(cors_list),
                    )
                )

                output_dict["accuracy_by_subject_and_level"][subject].append(
                    {
                        "level": level,
                        "num_correct": np.sum(cors_list),
                        "num_total": len(cors_list),
                        "accuracy": np.mean(cors_list),
                    }
                )

        print("#####################")
        f.write("#####################\n")
        for level in sorted(level_cors):
            if level not in level_cors.keys():
                print("Skipping", level)
                continue
            cors_list = level_cors[level]
            print(
                "Level {} Accuracy = {}/{} = {:.3f}".format(
                    level, np.sum(cors_list), len(cors_list), np.mean(cors_list)
                )
            )
            f.write(
                "Level {} Accuracy = {}/{} = {:.3f}\n".format(
                    level, np.sum(cors_list), len(cors_list), np.mean(cors_list)
                )
            )
            output_dict["accuracy_by_level"].append(
                {
                    "level": level,
                    "num_correct": np.sum(cors_list),
                    "num_total": len(cors_list),
                    "accuracy": np.mean(cors_list),
                }
            )

        print("#####################")
        f.write("#####################\n")
        for subject in [
            "Prealgebra",
            "Algebra",
            "Number Theory",
            "Counting & Probability",
            "Geometry",
            "Intermediate Algebra",
            "Precalculus",
        ]:
            if subject not in subject_cors.keys():
                print("Skipping", subject)
                continue
            cors_list = subject_cors[subject]
            print(
                "{} Accuracy = {}/{} = {:.3f}".format(
                    subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)
                )
            )
            f.write(
                "{} Accuracy = {}/{} = {:.3f}\n".format(
                    subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)
                )
            )
            output_dict["accuracy_by_subject"].append(
                {
                    "subject": subject,
                    "num_correct": np.sum(cors_list),
                    "num_total": len(cors_list),
                    "accuracy": np.mean(cors_list),
                }
            )
        print("#####################")
        f.write("#####################\n")
        total = len(all_problems)
        print(
            "Overall Accuracy = {}/{} = {:.3f}".format(
                correct, total, correct / total * 100
            )
        )
        f.write(
            "Overall Accuracy = {}/{} = {:.3f}\n".format(
                correct, total, correct / total * 100
            )
        )

        f.write(
            f"self-consistency rate: {num_consistency / total if args.inference_mode == 'self_consistency' else 'N/A'}\n"
        )

        output_dict["overall_accuracy"] = {
            "num_correct": correct,
            "num_total": total,
            "accuracy": correct / total,
        }

        class JSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.int64):
                    return int(obj)
                return super(JSONEncoder, self).default(obj)

        with open(os.path.join(args.save_dir, "results.json"), "w") as jf:
            json.dump(output_dict, jf, cls=JSONEncoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="")
    parser.add_argument("--save_dir", "-s", type=str, default="")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--inference_mode",
        "-i",
        type=str,
        default="normal",
        choices=[
            "normal",
            "self_consistency",
            "self_refinement",
            "adaptive_temp",
            "em_inf",
            "ice_self_consistency",
        ],
    )
    parser.add_argument("--n_trajs", type=int, default=4)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument(
        "--hyperparameters",
        type=str,
        default='{"threshold": 0.1, "kl_weight": 0.001, "learning_rate": 0.1, "n_grad_steps": 3}',
    )
    parser.add_argument("--num_processes", default=4, type=int)
    args = parser.parse_args()

    if args.inference_mode == "self_refinement":
        # Run self-refinement with the specified number of trajectories
        os.rmdir(args.save_dir)
        root_dir = os.path.dirname(args.save_dir)

        # initialize vllm
        vllm_model_instance = LLM(
            model=args.model,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype="bfloat16",
            enforce_eager=True,
            skip_tokenizer_init=False,
            gpu_memory_utilization=0.7,
        )

        for current_iter in range(args.n_trajs):
            args.save_dir = root_dir
            run(args, cur_iter=current_iter, vllm_model_instance=vllm_model_instance)
    else:
        run(args)
