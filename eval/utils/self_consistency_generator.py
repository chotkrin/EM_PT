import math
import os
import random
import time

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from vllm import LLM, SamplingParams

from .grader import math_equal
from .math_utils import grade_answer


def sc_inference(
    model_name,
    prompt_chunk,
    worker_id,
    max_new_tokens=3072,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    outputs = []
    prompt_generation_time = {}
    for i, prompt in tqdm(
        enumerate(prompt_chunk),
        total=len(prompt_chunk),
        desc=f"Processing prompts on Worker {worker_id} (process {os.getpid()})",
        position=worker_id,
        leave=True,
    ):
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)
        prompt_ids_len = input_ids["input_ids"].shape[-1]

        start_time = time.time()
        output_ids = model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
        )
        output = tokenizer.decode(
            output_ids[0][prompt_ids_len:],
            skip_special_tokens=True,
        )
        outputs.append(output)
        end_time = time.time()

        if prompt in prompt_generation_time:
            prompt_generation_time[prompt] = (
                prompt_generation_time[prompt] + end_time - start_time
            )
        else:
            prompt_generation_time[prompt] = end_time - start_time
    return outputs, prompt_generation_time


def self_consistency_output_selection(completions_dict, question_list):
    completions = []
    for i in range(len(question_list)):
        question = question_list[i]
        if question not in completions_dict:
            print(f"Question not found in completions_dict: {question}")
            continue

        outputs = completions_dict[question]
        if len(outputs) == 0:
            print(f"No outputs for question: {question}")
            continue

        answers = []
        for output in outputs:
            match_, extracted_answer = match_answer(output)
            if match_:
                answers.append(extracted_answer)
            else:
                answers.append(None)

        # Precompute equivalence matrix
        n = len(outputs)
        equivalence = [[False] * n for _ in range(n)]
        for j in range(n):
            for k in range(n):
                if j == k:
                    equivalence[j][k] = True
                elif answers[j] is None or answers[k] is None:
                    equivalence[j][k] = False
                else:
                    if grade_answer(answers[j], answers[k]):
                        equivalence[j][k] = True
                    else:
                        try:
                            if "\\pi" in answers[j] or "\\pi" in answers[k]:
                                equivs = []
                                for pi_val in [math.pi, 3.14]:
                                    equivs.append(
                                        math_equal(
                                            answers[j],
                                            answers[k],
                                            timeout=True,
                                            pi=pi_val,
                                        )
                                    )
                                equivalence[j][k] = any(equivs)
                            else:
                                equivalence[j][k] = math_equal(
                                    answers[j], answers[k], timeout=True
                                )
                        except Exception:
                            equivalence[j][k] = False

        # For each response, count how many others it matches
        match_counts = [sum(equivalence[i]) for i in range(n)]

        # Normalize and assign rewards
        rewards = [0] * len(outputs)
        for idx in range(len(match_counts)):
            rewards[idx] = match_counts[idx] / n

        # choose the highest and lowest reward output
        highest_reward = max(rewards)
        lowest_reward = min(rewards)

        # get all indices with the highest reward
        highest_indices = [idx for idx, x in enumerate(rewards) if x == highest_reward]
        completions.append(outputs[random.choice(highest_indices)])

    assert len(completions) == len(question_list)
    return completions


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


def generate_sample_batch_with_n_trajs(args, question_list, n_trajs):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="bfloat16",
        enforce_eager=True,
        skip_tokenizer_init=False,
        gpu_memory_utilization=0.7,
    )
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=args.temp,
        seed=42,
        best_of=1,
        stop=["\n###\nProblem: ", "<|eot_id|>"],
    )
    upsampled_question_list = question_list * n_trajs
    print(f"Sample prompt: {upsampled_question_list[0]}")
    outputs = llm.generate(upsampled_question_list, sampling_params, use_tqdm=True)

    assert len(outputs) == len(upsampled_question_list)

    completions = {}
    for question, output in zip(upsampled_question_list, outputs):
        if question not in completions:
            completions[question] = []
        output_str = output.outputs[0].text
        completions[question].append(output_str)

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
    ans_marker = "The answer is: "
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


def get_output_with_self_consistency(completions_dict, all_problems):
    completions = []
    num_consistency = 0
    for i in range(len(all_problems)):
        problem_data = all_problems[i]
        question = problem_data["question"]
        if question not in completions_dict:
            print(f"Question not found in completions_dict: {question}")
            continue

        outputs = completions_dict[question]
        if len(outputs) == 0:
            print(f"No outputs for question: {question}")
            continue

        # Get the most common answer
        answers = []
        for output in outputs:
            _, extracted_answer = match_answer(
                output.replace("<|im_end|>", "")
            )  # Assuming match_answer() exists
            answers.append(extracted_answer)

        # Step 2: Group responses based on the same answer
        answer_groups = defaultdict(list)
        for i, answer in enumerate(answers):
            answer_groups[answer].append(i)

        # Step 3: Assign rewards based on group size
        total_responses = len(outputs)
        rewards = []

        for i in range(total_responses):
            answer = answers[i]
            group_size = len(answer_groups[answer])
            reward = group_size / total_responses
            rewards.append(reward)

        # Step 4: Select chosen and rejected responses
        # Find indices of highest reward
        max_reward = max(rewards)
        min_reward = min(rewards)
        if max_reward != min_reward:
            num_consistency += 1

        max_reward_indices = [i for i, r in enumerate(rewards) if r == max_reward]

        # Randomly choose one from each group
        chosen_idx = random.choice(max_reward_indices)

        completions.append(outputs[chosen_idx])

    assert len(completions) == len(all_problems)
    return completions, num_consistency / len(all_problems)
