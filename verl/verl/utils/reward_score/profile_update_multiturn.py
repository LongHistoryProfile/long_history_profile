import numpy as np
import torch
from typing import List, Dict, Any

from verl.protocol import DataProto
from verl.utils.reward_score.duet import profile_controller_compute_reward


# def unwrap_np_obj(x):
#     if isinstance(x, np.ndarray) and x.dtype == object:
#         return x.item()
#     return x


def _get_targets_list(non_tensor_batch: dict, i: int) -> List[dict]:
    rm = non_tensor_batch.get("reward_model", None)
    if rm is None:
        return []
    rm_i = rm[i]
    gt = rm_i.get("ground_truth", {})
    targets = gt.get("targets", [])

    return targets if isinstance(targets, list) else []


def _extract_turn_solutions(non_tensor_batch: dict, i: int) -> List[str]:
    """
    必须从 rollout 后的 messages 中取每轮 assistant 的输出。
    这取决于 verl 版本，优先 rollout_messages。
    """
    for k in ["rollout_messages", "messages", "full_messages", "conversation"]:
        if k in non_tensor_batch:
            arr = non_tensor_batch[k]
            if len(arr) <= i:
                continue
            msg_list = arr[i]
            if not isinstance(msg_list, list):
                continue
            sols = []
            for m in msg_list:
                if isinstance(m, dict) and m.get("role") == "assistant":
                    sols.append(str(m.get("content", "")))
            if sols:
                return sols
    return []


def _get_turn_response_lens(non_tensor_batch: dict, i: int) -> List[int]:
    """
    强烈建议你在 rollout 阶段把它写出来，否则无法精确把 reward_t 对齐到 turn t 的 token。
    """
    if "turn_response_lens" not in non_tensor_batch:
        return []
    lens_i = non_tensor_batch["turn_response_lens"][i]
    if isinstance(lens_i, list):
        return [int(x) for x in lens_i]
    return []


def multiturn_profile_reward(data: DataProto, **kwargs) -> Dict[str, Any]:
    """
    verl custom_reward_function 入口。
    输出必须包含 reward_tensor (token-level) 才能 PPO 正确训练。
    """
    batch = data.batch
    non_tensor = data.non_tensor_batch

    input_ids = batch["input_ids"]
    device = input_ids.device
    B, S = input_ids.shape

    reward_tensor = torch.zeros((B, S), dtype=torch.float32, device=device)

    # prompts length for alignment
    assert "prompts" in batch, "Need batch['prompts'] to locate response token offsets"
    prompt_len = batch["prompts"].shape[1]

    # response_mask align response tokens in full seq
    response_mask = batch.get("response_mask", None)
    if response_mask is None:
        # fallback to attention_mask after prompt
        response_mask = batch["attention_mask"][:, prompt_len:]

    # debug extras
    extra = {
        "turn_rewards": [],
        "n_turns": [],
    }

    for i in range(B):
        targets = _get_targets_list(non_tensor, i)
        sols = _extract_turn_solutions(non_tensor, i)

        n_turns = min(len(targets), len(sols))
        if n_turns == 0:
            extra["turn_rewards"].append([])
            extra["n_turns"].append(0)
            continue

        turn_rewards = []
        for t in range(n_turns):
            out = profile_controller_compute_reward(
                data_source="ProfileControllerMultiTurn",
                solution_str=sols[t],
                ground_truth={"target_review": targets[t]},
                extra_info=None,
            )
            turn_rewards.append(float(out["score"]))

        # ---- assign reward to tokens (per turn) ----
        turn_lens = _get_turn_response_lens(non_tensor, i)

        if len(turn_lens) >= n_turns:
            # precise per-turn assignment
            cum = 0
            for t in range(n_turns):
                L = int(turn_lens[t])
                if L <= 0:
                    continue
                last_tok = cum + L - 1
                reward_tensor[i, prompt_len + last_tok] = turn_rewards[t]
                cum += L
        else:
            # fallback: assign only terminal reward to last response token
            idx = torch.where(response_mask[i].bool())[0]
            if len(idx) > 0:
                reward_tensor[i, prompt_len + idx[-1]] = turn_rewards[-1]

        extra["turn_rewards"].append(turn_rewards)
        extra["n_turns"].append(n_turns)

    return {
        "reward_tensor": reward_tensor,
        "reward_extra_info": extra,
    }
