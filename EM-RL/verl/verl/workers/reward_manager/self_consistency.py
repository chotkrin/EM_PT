# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict, Counter
import math
from verl.utils.reward_score.prime_math import grade_answer
from verl.utils.reward_score.prime_math.grader import math_equal

class SelfConsistencyRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score


    def _last_boxed_only_string(self, string):
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


    def match_answer(self, response):
        is_matched = False
        ans_marker = "The answer is: "
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker) :].strip()
            if response.endswith("\n"):
                response = response[:-2]
        
        ans_marker = "The final answer is: "
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

        # # Find boxed
        ans_boxed = self._last_boxed_only_string(response)
        if ans_boxed:
            is_matched = True
            response = ans_boxed

        # Grade
        return is_matched, response


    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
        # raise NotImplementedError

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # print(len(data))
        # print(data.batch['responses'].shape)
        # print(data.batch['prompts'].shape)
        # input("Press Enter to continue...")
        # print(data.batch.keys())
        # print(data.non_tensor_batch.keys())
        # print(data.non_tensor_batch['extra_info'])
        # print(data.non_tensor_batch['index'])
        # print(data.meta_info)
        # raise NotImplementedError
        already_print_data_sources = {}
        prompt_to_indices = defaultdict(list)
        # for idx, prompt in enumerate(prompts):
        #     prompt_to_indices[prompt].append(idx)

        prompt_li = []
        responses_li = []
        
        for i in range(len(data)):
            # if i > 16: 
            #     raise NotImplementedError
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            # print(f'prompt_str {prompt_str}')
            prompt_li.append(prompt_str)
            responses_li.append(response_str)
            prompt_to_indices[prompt_str.strip()].append(i)
        
        rewards = [0]*len(responses_li)
        for prompt, indices in prompt_to_indices.items():
        # Extract all answers for this prompt
            all_extracted = []
            for i in indices:
                match_, ans = self.match_answer(responses_li[i])
                if match_:
                    all_extracted.append(ans)
                else:
                    all_extracted.append(None)
                # all_extracted = [self.extract_answer(responses_li[i]) for i in indices]

            # Precompute equivalence matrix
            n = len(indices)
            # print(f'n {n}')
            equivalence = [[False]*n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        equivalence[i][j] = True
                    elif all_extracted[i] is None or all_extracted[j] is None:
                        equivalence[i][j] = False
                    else:
                        if grade_answer(all_extracted[i], all_extracted[j]):
                            equivalence[i][j] = True
                        else:
                            try:
                                if "\\pi" in all_extracted[i] or "\\pi" in all_extracted[j]:
                                    equivs = []
                                    for pi_val in [math.pi, 3.14]:
                                        equivs.append(math_equal(all_extracted[i], all_extracted[j], timeout=True, pi=pi_val))
                                    equivalence[i][j] = any(equivs)
                                else:
                                    equivalence[i][j] = math_equal(all_extracted[i], all_extracted[j], timeout=True)
                            except Exception:
                                equivalence[i][j] = False

            # For each response, count how many others it matches
            match_counts = [sum(equivalence[i]) for i in range(n)]

            # Normalize and assign rewards
            for idx_in_list, idx_in_batch in enumerate(indices):
                # reward_tensor[idx_in_batch, valid_response_length - 1] = score
                rewards[idx_in_batch] = match_counts[idx_in_list] / n

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            reward_tensor[i, valid_response_length - 1] = rewards[i]
            # print(rewards[i])

 
           
        # print(math.mean(rewards))
        return reward_tensor
