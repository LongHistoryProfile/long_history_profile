import sys
sys.path.append("/home/aiscuser/yifeisun/project_1128/verl")
from prompt_config import (
    PROFILE_CONTROLLER_RATING_SYSTEM_PROMPT,
    PROFILE_CONTROLLER_RATING_PREDICTOR_PROMPT
)
import re
import requests, json
def call(system_prompt, user_prompt,seed=42):
    api = "http://0.0.0.0:8008/chat"

    payload = {
        "system": system_prompt,
        "user": user_prompt,
        "max_tokens": 16,
        "thinking": False,
        "seed":seed
    }

    res = requests.post(api, json=payload)
    text = res.json()["response"]
    return text
def extract_user_profile(completion_text: str) -> tuple[str, str]:
    # 先找最后一个<user_profile>，再找其后最近的</user_profile>
    user_profile = None

    user_start = completion_text.lower().rfind('<user_profile>')
    if user_start != -1:
        user_end = completion_text.lower().find('</user_profile>', user_start)
        if user_end != -1:
            user_profile = completion_text[user_start+14:user_end].strip()
        else:
            print("No closing </user_profile> tag found after last <user_profile>.")
    else:
        print("No <user_profile> tag found.")

    return user_profile 

def extract_reason_user_profile(completion_text: str) -> tuple[str | None, str | None]:
    """
    Extract the LAST <reason> and <user_profile> blocks.
    Returns (reason, user_profile)
    """

    text_lower = completion_text.lower()

    # ---------- Extract <reason> ----------
    reason = None
    reason_start = text_lower.rfind('<reason>')
    if reason_start != -1:
        reason_end = text_lower.find('</reason>', reason_start)
        if reason_end != -1:
            reason = completion_text[reason_start + len('<reason>'):reason_end].strip()
        else:
            print("No closing </reason> tag found after last <reason>.")
    else:
        print("No <reason> tag found.")

    # ---------- Extract <user_profile> ----------
    user_profile = None
    user_start = text_lower.rfind('<user_profile>')
    if user_start != -1:
        user_end = text_lower.find('</user_profile>', user_start)
        if user_end != -1:
            user_profile = completion_text[user_start + len('<user_profile>'):user_end].strip()
        else:
            print("No closing </user_profile> tag found after last <user_profile>.")
    else:
        print("No <user_profile> tag found.")

    return reason, user_profile


RATING_TAG_PATTERN = re.compile(
    r"<rating>\s*((?:[1-4]\.\d{2}|5\.00))\s*</rating>",
    re.DOTALL
)

def extract_rating(rating_text: str) -> tuple[float, bool]:
    matches = list(RATING_TAG_PATTERN.finditer(rating_text))
    if not matches:
        print("No valid <rating> tag found.")
        return 0.0, False

    tag_content = matches[-1].group(1).strip()
    try:
        return float(tag_content), True
    except Exception:
        return 0.0, False


def profile_controller_compute_reward(data_source, solution_str, ground_truth, extra_info=None):
    reviews=ground_truth["reviews"]
    start_idx=ground_truth["start_idx"]

    target_review=reviews[start_idx]


    reason, user_profile=extract_reason_user_profile(solution_str)
    total_reward=0

    predicted_rating = -1 # means error
    gt_rating = target_review["rating"]

    if reason is None or user_profile is None:
        print("Profile extraction failed.....")
        format_reward = -1.0
        total_reward = format_reward
    else:
        # 构造评分预测prompt
        rating_prompt= PROFILE_CONTROLLER_RATING_PREDICTOR_PROMPT.format(
            user_profile=solution_str,
            item_title=target_review["item_title"],
            item_description=target_review["description"][:3000],
            item_avg_rating=target_review["item_average_rating"]
        )

        rating_text= call(
            PROFILE_CONTROLLER_RATING_SYSTEM_PROMPT,
            rating_prompt
        )
        predicted_rating, format_valid = extract_rating(rating_text)
        total_reward=0
        
        if format_valid is False:
            format_reward = -1.0
            total_reward = format_reward
        else:
            # 计算准确奖励
            max_possible_reward = 1.0
            error = abs(predicted_rating - gt_rating)/4.0  # 归一化误差到[0,1]
            acc_reward = max_possible_reward - error
            
            # 总奖励 = 准确奖励
            total_reward = acc_reward
    return {
        "score": total_reward,
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "predicted_rating": predicted_rating,
        "gt_rating": gt_rating,
        "reason":reason,
        "user_profile":user_profile
    }

