import re
import torch
from reasoning_gym import get_score_answer_fn
import numpy as np

FORMAT_REWARD_WEIGHT = 0.15
CORRECTNESS_REWARD_WEIGHT = 0.85


def extract_answer(response):
    answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer:
        return answer.group(1).strip()
    return ""

def calculate_format_reward(response):
    if ("think>" not in response) and ("<answer>" not in response) and ("</think>" not in response) and ("</answer>" not in response):
        return -1  # Penalty for not following format
    format_reward = 0
    if "<think>" in response:
        format_reward += 0.15
    if "</think>" in response:
        format_reward += 0.15
    if "<answer>" in response and "</answer>" in response:
        format_reward += 0.7
    return format_reward

def calculate_logits(model, responses, attention_mask):
    logits = model(input_ids=responses, attention_mask=attention_mask).logits
    log_probs = torch.log_softmax(logits, dim=-1)

    selected_log_probs = torch.gather(input=log_probs,
                                      dim=2,
                                      index = responses.unsqueeze(-1)
                                      ).squeeze(-1)
    return selected_log_probs

def calculate_correctness_reward(response, validation_object):
    score_fn = get_score_answer_fn(validation_object['metadata']['source_dataset'])
    return score_fn(response, validation_object)

def calculate_total_reward(batch_response, validation_objects):
    format_rewards = np.array([calculate_format_reward(resp) for resp in batch_response])
    correctness_rewards = np.array([calculate_correctness_reward(extract_answer(resp), val_obj) for resp, val_obj in zip(batch_response, validation_objects)])
    total_rewards = (FORMAT_REWARD_WEIGHT * format_rewards) + (CORRECTNESS_REWARD_WEIGHT * correctness_rewards)
    return total_rewards