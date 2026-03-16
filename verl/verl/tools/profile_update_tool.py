import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import requests

from verl.utils.rollout_trace import rollout_trace_op
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

import sys
sys.path.append("/home/aiscuser/yifeisun/project_1128/verl")
from prompt_config import (
    PROFILE_CONTROLLER_USER_PROMPT,
    PROFILE_CONTROLLER_RATING_SYSTEM_PROMPT,
    PROFILE_CONTROLLER_RATING_PREDICTOR_PROMPT,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# -------------------------
# helpers
# -------------------------
# def unwrap_np_obj(x):
#     """unwrap numpy object array to python object if needed."""
#     if isinstance(x, np.ndarray) and x.dtype == object:
#         return x.item()
#     return x


def build_review_sentence(r: dict) -> str:
    return (
        f"Item: {r['item_title']} "
        f"Rating: {float(r['rating'])} "
        f"(item_avg_rating={float(r['item_average_rating'])}, item_rating_number={int(r['item_rating_number'])})\n"
        f"Review: {r['title']}: {str(r['text'])[:200].strip()}"
    )


def build_recent_reviews_block(reviews: list[dict]) -> str:
    review_sentences = [build_review_sentence(r) for r in reviews]
    return "\n".join([f"[Review {i+1}] {review_sentences[i]}" for i in range(len(reviews))])


def build_user_prompt(existing_profile: str, window_reviews: list[dict], t: int) -> str:
    recent_reviews = build_recent_reviews_block(window_reviews)
    prompt = PROFILE_CONTROLLER_USER_PROMPT.format(
        existing_profile=existing_profile,
        recent_reviews=recent_reviews,
    )
    # 可选：给 prompt 加 turn marker，便于 debug
    return  prompt


# -------------------------
# reward oracle
# -------------------------
def call_llm(system_prompt: str, user_prompt: str, seed: int = 42) -> str:
    """
    Your local reward oracle (Qwen3-8B server).
    """
    api = "http://0.0.0.0:8008/chat"
    payload = {
        "system": system_prompt,
        "user": user_prompt,
        "max_tokens": 16,
        "thinking": False,
        "seed": seed,
    }
    res = requests.post(api, json=payload, timeout=120)
    res.raise_for_status()
    return res.json()["response"]


def extract_user_profile(completion_text: str) -> Optional[str]:
    """
    Extract last <user_profile>...</user_profile>
    """
    if not isinstance(completion_text, str):
        completion_text = str(completion_text)

    lower = completion_text.lower()
    user_start = lower.rfind("<user_profile>")
    if user_start == -1:
        return None
    user_end = lower.find("</user_profile>", user_start)
    if user_end == -1:
        return None
    return completion_text[user_start + len("<user_profile>") : user_end].strip()


RATING_TAG_PATTERN = re.compile(
    r"<rating>\s*((?:[1-4]\.\d{2}|5\.00))\s*</rating>",
    re.DOTALL,
)


def extract_rating(rating_text: str) -> tuple[float, bool]:
    matches = list(RATING_TAG_PATTERN.finditer(rating_text))
    if not matches:
        return 0.0, False

    tag_content = matches[-1].group(1).strip()
    try:
        return float(tag_content), True
    except Exception:
        return 0.0, False


def compute_profile_step_reward(solution_str: str, target_review: dict) -> dict:
    """
    Return dict consistent with verl reward_score format.
    """
    user_profile = extract_user_profile(solution_str)

    predicted_rating = -1.0
    gt_rating = float(target_review["rating"])

    if user_profile is None:
        total_reward = -1.0
        return {
            "score": total_reward,
            "predicted_rating": predicted_rating,
            "gt_rating": gt_rating,
            "format_valid": False,
        }

    rating_prompt = PROFILE_CONTROLLER_RATING_PREDICTOR_PROMPT.format(
        user_profile=solution_str,
        item_title=target_review["item_title"],
        item_description=str(target_review["description"])[:3000],
        item_avg_rating=float(target_review["item_average_rating"]),
    )
    rating_text = call_llm(PROFILE_CONTROLLER_RATING_SYSTEM_PROMPT, rating_prompt)

    predicted_rating, format_valid = extract_rating(rating_text)
    if not format_valid:
        total_reward = -1.0
    else:
        # normalized abs error, max reward=1
        err = abs(float(predicted_rating) - gt_rating) / 4.0
        total_reward = 1.0 - err

    return {
        "score": float(total_reward),
        "predicted_rating": float(predicted_rating) if format_valid else -1.0,
        "gt_rating": float(gt_rating),
        "format_valid": bool(format_valid),
    }


# -------------------------
# Tool implementation
# -------------------------
class SlidingProfileUpdateTool(BaseTool):
    """
    GSM8K-style multi-turn tool:

    - create(): init state, return first prompt as ToolResponse(text=...)
    - execute(): get profile, compute step reward using current target_review,
                return next prompt as ToolResponse and step tool_reward

    Dataset must pass into create_kwargs:
        {
          "user_id": ...,
          "init_profile": ...,
          "windows": [...],   # list[list[review dict]]
          "targets": [...],   # list[review dict], same length as windows
        }
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Example tool schema (YAML):
        {
          "type": "function",
          "function": {
            "name": "update_profile",
            "description": "Update user profile given recent reviews. Return updated profile in <user_profile> tags.",
            "parameters": {
              "type": "object",
              "properties": {
                "profile": {
                  "type": "string",
                  "description": "The updated user profile, must be wrapped with <user_profile>...</user_profile>."
                }
              },
              "required": ["profile"]
            }
          }
        }
        """
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}

        # safety
        self.max_turns_safety = int(config.get("max_turns_safety", 64))

        # shaping, like gsm8k: non-improvement penalty
        self.enable_improve_penalty = bool(config.get("enable_improve_penalty", True))
        self.improve_penalty = float(config.get("improve_penalty", -0.05))

        # oracle
        self.reward_seed = int(config.get("reward_seed", 42))
        print(f"==================================init SlidingProfileUpdateTool, config={config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """
        Must return (instance_id, ToolResponse) exactly like gsm8k.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs =  kwargs.get("create_kwargs", {})

        user_id = create_kwargs.get("user_id", None)
        init_profile = create_kwargs.get("init_profile", "")
        windows =  create_kwargs.get("windows", None)
        targets =  create_kwargs.get("targets", None)

        if windows is None:
            raise ValueError("SlidingProfileUpdateTool requires create_kwargs['windows']")
        if targets is None:
            raise ValueError("SlidingProfileUpdateTool requires create_kwargs['targets']")

        if not isinstance(windows, list) or len(windows) == 0:
            raise ValueError("windows must be non-empty list")
        if not isinstance(targets, list) or len(targets) == 0:
            raise ValueError("targets must be non-empty list")
        if len(windows) != len(targets):
            raise ValueError(f"len(windows)={len(windows)} must equal len(targets)={len(targets)}")

        # safety cap
        if len(windows) > self.max_turns_safety:
            windows = windows[: self.max_turns_safety]
            targets = targets[: self.max_turns_safety]

        self._instance_dict[instance_id] = {
            "user_id": user_id,
            "t": 0,
            "windows": windows,
            "targets": targets,
            "profile": init_profile,
            "profiles": [],
            "step_rewards": [],
            "last_score": None,
        }

        first_prompt = build_user_prompt(init_profile, windows[0], t=0)
        print("========== SlidingProfileUpdateTool.create(): first prompt ==========")
        print(first_prompt)

        return instance_id, ToolResponse(text=first_prompt)

    async def calc_reward(self, instance_id: str, solution_str: str) -> dict:
        """
        GSM8K tool has calc_reward() which returns float.
        Here we return a dict extra info for logging, but caller uses ["score"].
        """
        inst = self._instance_dict[instance_id]
        t = int(inst["t"])
        target_review = inst["targets"][t]

        out = compute_profile_step_reward(solution_str=solution_str, target_review=target_review)
        return out

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """
        Must return (ToolResponse, tool_reward: float, extra_info: dict)
        exactly like gsm8k.
        """
        inst = self._instance_dict[instance_id]

        profile = parameters.get("profile", "")
        if not isinstance(profile, str):
            profile = str(profile)

        # guard
        t = int(inst["t"])
        if t >= len(inst["targets"]):
            return ToolResponse(text="Episode done."), 0.0, {"done": True, "n_turns": len(inst["windows"])}

        # --- compute reward ---
        reward_out = await self.calc_reward(instance_id, solution_str=profile)
        score = float(reward_out["score"])
        tool_reward = score

        # shaping: penalize non-improvement
        if self.enable_improve_penalty and inst["last_score"] is not None:
            if score <= float(inst["last_score"]):
                tool_reward += float(self.improve_penalty)

        inst["last_score"] = score
        inst["profiles"].append(profile)
        inst["profile"] = profile
        inst["step_rewards"].append(float(tool_reward))
        inst["t"] += 1

        # done?
        done = inst["t"] >= len(inst["windows"])
        if done:
            return ToolResponse(text="Episode done."), float(tool_reward), {
                "done": True,
                "n_turns": len(inst["windows"]),
                "t": inst["t"],
                "score": score,
                "tool_reward": tool_reward,
                "predicted_rating": reward_out.get("predicted_rating", -1.0),
                "gt_rating": reward_out.get("gt_rating", -1.0),
                "format_valid": reward_out.get("format_valid", False),
                "step_rewards": inst["step_rewards"],
            }

        # next prompt
        next_t = int(inst["t"])
        next_prompt = build_user_prompt(inst["profile"], inst["windows"][next_t], t=next_t)

        print(f"========== SlidingProfileUpdateTool.execute(): next prompt t={next_t} ==========")
        print(next_prompt)

        return ToolResponse(text=next_prompt), float(tool_reward), {
            "done": False,
            "t": next_t,
            "score": score,
            "tool_reward": tool_reward,
            "predicted_rating": reward_out.get("predicted_rating", -1.0),
            "gt_rating": reward_out.get("gt_rating", -1.0),
            "format_valid": reward_out.get("format_valid", False),
            "step_rewards": inst["step_rewards"],
        }

    async def release(self, instance_id: str, **kwargs) -> None:
        """
        Must cleanup instance like gsm8k.
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
