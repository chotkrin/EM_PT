import random

from judge import Judger
from utils import *

ugphysics_judger = Judger(strict_extract=True, judge_model="Qwen/Qwen2.5-7B-Instruct")


def output_selection_with_self_consistency(output_list, precision=1e-8):
    """
    Check the correctness for single item.

    Parameters:
        data_item: Data to be checked.
            data format:

    Returns:
        correctness: A list that contains the correctness for all prompts.
    """
    n = len(output_list)
    equivalence = [[False] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            correctness = ugphysics_judger.auto_judge_for_self_consistency(
                output_list[i], output_list[j], precision=precision
            )
            if correctness:
                equivalence[i][j] = True
            else:
                equivalence[i][j] = False

    # For each response, count how many others it matches
    match_counts = [sum(equivalence[i]) for i in range(n)]

    # Normalize and assign rewards
    rewards = [0] * len(output_list)
    for idx in range(len(match_counts)):
        rewards[idx] = match_counts[idx] / n

    # choose the highest and lowest reward output
    highest_reward = max(rewards)

    # get all indices with the highest reward
    highest_indices = [idx for idx, x in enumerate(rewards) if x == highest_reward]

    return output_list[random.choice(highest_indices)]
