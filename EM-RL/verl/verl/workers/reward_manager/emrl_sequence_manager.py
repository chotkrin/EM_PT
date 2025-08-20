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

class EMRLSequenceRewardManager:
    """The reward manager.
    """

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

    def negative_entropy_from_log_probs(self, log_probs: torch.Tensor):
        """Compute negative entropy given log probabilities."""
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return -entropy  # Negative entropy to minimize it
    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # print('PROB REWARDER NO CONSISTENCY')
        

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
    
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]


            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            joint_prob = data_item.batch['old_log_probs'][:valid_response_length].mean()
            reward_tensor[i, valid_response_length - 1] = joint_prob

 
           
        return reward_tensor
