import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from parser import *

import numpy as np
import torch

from data_loader import load_data
from evaluate import evaluate
from model_utils import generate_completions, load_hf_lm_and_tokenizer
from python_executor import PythonExecutor
from tqdm import tqdm
from trajectory import *
from transformers import AutoTokenizer

from general_utils import construct_prompt, load_jsonl, save_jsonl, set_seed
from vllm import LLM, SamplingParams

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.em_inf_utils import (
    min_entropy_inference,
    AdaptiveTemperatureProcessor
)
from utils.self_consistency_generator import (
    generate_sample_batch_with_n_trajs,
    self_consistency_output_selection,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="../../../data", type=str)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--inference_mode", "-i", type=str, default="normal", choices=["normal", "self_consistency", "self_refinement", "adaptive_temp", "em_inf", "ice_self_consistency"])
    parser.add_argument("--n_trajs", type=int, default=4)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=4096, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument(
        "--hyperparameters",
        default='{"threshold": 0.3, "kl_weight": 0.01, "learning_rate": 0.05, "n_grad_steps": 3}',
        type=str,
    )
    parser.add_argument("--num_processes", default=4, type=int)
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    if args.inference_mode == "em_inf":
        args.use_vllm = False
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = (
        f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    )
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(list(load_jsonl(f"{output_dir}/{data_name}/{f}")))

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
        if (
            args.inference_mode == "normal"
            or args.inference_mode == "self_consistency"
            or args.inference_mode == "self_refinement"
        ):
            llm = LLM(
                model=args.model_name_or_path,
                tensor_parallel_size=4,
                pipeline_parallel_size=args.pipeline_parallel_size,
                trust_remote_code=True,
                gpu_memory_utilization=0.7,
            )

            tokenizer = None
            if args.apply_chat_template:
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model_name_or_path, trust_remote_code=True
                )
    else:
        if args.inference_mode == "em_inf":
            llm = None
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
        else:
            llm, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path,
                load_in_half=True,
                use_fast_tokenizer=True,
                use_safetensors=args.use_safetensors,
            )

    # infer & eval
    if args.inference_mode == "self_refinement":
        for current_iter in range(3):
            print(f"Self-refinement iteration {current_iter}")
            data_list = args.data_names.split(",")
            results = []
            for data_name in data_list:
                results.append(
                    main(llm, tokenizer, data_name, args, cur_iter=current_iter)
                )

            # add "avg" result to data_list and results
            data_list.append("avg")
            results.append(
                {
                    "acc": sum([result["acc"] for result in results]) / len(results),
                }
            )

            # print all results
            pad = max([len(data_name) for data_name in data_list])
            print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
            print(
                "\t".join(
                    [f"{result['acc']:.1f}".ljust(pad, " ") for result in results]
                )
            )
    else:
        data_list = args.data_names.split(",")
        results = []
        for data_name in data_list:
            results.append(main(llm, tokenizer, data_name, args))

        # add "avg" result to data_list and results
        data_list.append("avg")
        results.append(
            {
                "acc": sum([result["acc"] for result in results]) / len(results),
            }
        )

        # print all results
        pad = max([len(data_name) for data_name in data_list])
        print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
        print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(llm, tokenizer, data_name, args, cur_iter=0):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)
        if idx == args.start:
            print("========before process")
            print(full_prompt)
        # 提取 <|im_start|>user 和 <|im_end|>之间的内容为question
        # 正则表达式
        pattern = r"<\|im_start\|>user(.*?)<\|im_end\|>"
        # 提取内容
        matches = re.findall(pattern, full_prompt, re.DOTALL)
        full_prompt = matches[0].strip()
        if idx == args.start:
            print("========after process")
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]

    system_prompt = open("system_prompt.md").read()
    system_prompt = system_prompt.replace("\n", "")

    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt.strip()},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # start inference
    for epoch in range(max_func_call):
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        print("-" * 20, "Epoch", epoch)
        # get all outputs
        prompts = [item[1] for item in current_prompts]
        if epoch == 0:
            print("========before llm generate")
            print(prompts[0])
        if args.use_vllm:
            if args.inference_mode == "normal":
                outputs = llm.generate(
                    prompts,
                    SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens_per_call,
                        seed=42,
                        n=1,
                        stop=stop_words,
                        stop_token_ids=(
                            [151645, 151643]
                            if "qwen2" in args.model_name_or_path.lower()
                            else None
                        ),
                    ),
                )

                outputs = sorted(
                    outputs, key=lambda x: int(x.request_id)
                )  # sort outputs by request_id
                outputs = [output.outputs[0].text for output in outputs]
                
            elif args.inference_mode == "adaptive_temp":
                hyperparameters = json.loads(args.hyperparameters)
                print(f"Running with adaptive temperature inference mode using hyperparameters {hyperparameters}")
                
                logits_processor = AdaptiveTemperatureProcessor(tmax=hyperparameters["tmax"], tmin=hyperparameters["tmin"],max_iter=hyperparameters["max_iter"], tol=hyperparameters["tol"], target_ratio=hyperparameters["target_ratio"], target_threshold=hyperparameters["threshold"])
                sampling_params = SamplingParams(
                    max_tokens=4096,
                    temperature=1.0,
                    seed=42,
                    logits_processors=[logits_processor],  
                    stop=stop_words,
                    stop_token_ids=(
                        [151645, 151643]
                        if "qwen2" in args.model_name_or_path.lower()
                        else None
                    ),
                )

                vllm_outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
                outputs = [output.outputs[0].text for output in vllm_outputs]
                
                
            elif args.inference_mode == "self_consistency":
                upsampled_prompts = prompts * args.n_trajs
                upsampled_outputs = llm.generate(
                    upsampled_prompts,
                    SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens_per_call,
                        n=1,
                        stop=stop_words,
                        stop_token_ids=(
                            [151645, 151643]
                            if "qwen2" in args.model_name_or_path.lower()
                            else None
                        ),
                    ),
                )
                completions = {}
                assert len(upsampled_outputs) == len(upsampled_prompts)
                for question, output in zip(upsampled_prompts, upsampled_outputs):
                    if question not in completions:
                        completions[question] = []
                    completions[question].append(output.outputs[0].text)

                outputs = self_consistency_output_selection(
                    completions, prompts
                )

            elif args.inference_mode == "self_refinement":
                print("*" * 30)
                print(
                    f"Generating with self-refinement inference mode: iteration {cur_iter}"
                )

                root_dir = os.path.dirname(args.output_dir)
                out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
                out_file = f"{root_dir}/iter_{cur_iter}/qwen_math/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"

                # get previous iteration's prompt and completions if current_iter > 0:
                current_prompt_prefix = [""] * len(prompts)
                if cur_iter > 0:
                    print("Loading previous iteration's prompts...")

                    previous_iter_dir = os.path.join(
                        root_dir,
                        f"iter_{cur_iter - 1}",
                        "qwen_math",
                        data_name,
                    )

                    # read previous iteration's concatenated prompts
                    current_prompt_prefix = []
                    with open(
                        os.path.join(
                            previous_iter_dir, "self_refinement_prompts.jsonl"
                        ),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        for line in f:
                            data = json.loads(line)
                            current_prompt_prefix.append(data["concatenated_prompt"])

                current_question_list = []
                for prefix, question in zip(current_prompt_prefix, prompts):
                    current_question_list.append(prefix + question)

                # completion
                outputs = llm.generate(
                    current_question_list,
                    SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens_per_call,
                        seed=42,
                        n=1,
                        stop=stop_words,
                        stop_token_ids=(
                            [151645, 151643]
                            if "qwen2" in args.model_name_or_path.lower()
                            else None
                        ),
                    ),
                )

                outputs = sorted(
                    outputs, key=lambda x: int(x.request_id)
                )  # sort outputs by request_id
                outputs = [output.outputs[0].text for output in outputs]

                # set current iteration's saving path
                save_dir = os.path.join(
                    root_dir,
                    f"iter_{cur_iter}",
                    "qwen_math",
                    data_name,
                )
                os.makedirs(save_dir, exist_ok=True)
                print(f"current iteration's saving path: {save_dir}")

                # save current iteration's prompts and completions together into a jsonl file, where each line is a dict: {"id": id, "prompt": prompt}
                tmp_data = []
                for problem_data, current_prompt, model_output in zip(
                    prompts, current_question_list, outputs
                ):
                    tmp_data.append(
                        {
                            "original_prompt": problem_data,
                            "concatenated_prompt": current_prompt
                            + model_output
                            + "\n\n",
                        }
                    )

                with open(
                    os.path.join(save_dir, "self_refinement_prompts.jsonl"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    for line in tmp_data:
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
        else:
            if args.inference_mode == "em_inf":
                import multiprocessing as mp

                ################################# Data Parallelism Inference setup #################################
                try:
                    mp.set_start_method("spawn")  # Important for CUDA safety!
                except RuntimeError:
                    print("Spawn method already set. Continuing...")
                num_processes = args.num_processes
                
                torch.manual_seed(42)
                np.random.seed(42)
                random.seed(42)
                ####################################################################################################

                # --- 1. Prepare and Chunkify prompts ---
                k, m = divmod(len(prompts), num_processes)
                prompt_chunks = [
                    prompts[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
                    for i in range(num_processes)
                ]

                # --- 2. Create and Start Processes ---
                hyperparameters = json.loads(args.hyperparameters)     
                print(f"Running with EM-INF inference mode using hyperparameters {hyperparameters}")

                with mp.Pool(processes=num_processes) as pool:
                    results_list = pool.starmap(
                        min_entropy_inference,
                        [
                            (args.model_name_or_path, chunk, i, args.temperature, hyperparameters)
                            for i, chunk in enumerate(prompt_chunks)
                        ],
                    )

                outputs = [
                    output for sublist in results_list for output in sublist
                ]

                # --- 3. Verification and Cleanup ---
                if len(outputs) == len(prompts):
                    print("All completions generated successfully.")
                else:
                    print("Missing completions detected.")

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )
    
    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    if args.inference_mode == "em_inf":
        threshold_str = str(hyperparameters["threshold"])
        kl_weight_str = str(hyperparameters["kl_weight"])
        grad_steps_str = str(hyperparameters["n_grad_steps"])
        lr_str = str(hyperparameters["learning_rate"])
        outfile = out_file.replace(
            ".jsonl",
            f"_{args.prompt_type}_threshold_{threshold_str}_klweight_{kl_weight_str}_gradsteps_{grad_steps_str}_lr_{lr_str}_metrics.json",
        )
    elif args.inference_mode == "adaptive_temp":
        ratio_str = str(hyperparameters["target_ratio"])
        threshold_str = str(hyperparameters["threshold"])
        outfile = out_file.replace(
            ".jsonl",
            f"_{args.prompt_type}_threshold_{threshold_str}_ratio_{ratio_str}_metrics.json",
        )
    else:
        outfile = out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json")

    with open(
        outfile, "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
