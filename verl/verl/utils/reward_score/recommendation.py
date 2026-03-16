import requests, json
from prompt_config import (
    MOVIE_PREDICT_MOOD_SYSTEM_PROMPT,
    MOVIE_PREDICT_MOOD_USER_PROMPT,
    MOVIE_SENTIMENT_PREDICTION_SYSTEM_PROMPT,
    MOVIE_SENTIMENT_PREDICTION_USER_PROMPT,
    MOVIE_PROFILE_SENTIMENT_PREDICTION_SYSTEM_PROMPT,
    MOVIE_PROFILE_SENTIMENT_PREDICTION_USER_PROMPT,
    MOVIE_SESSION_ITEM_TEMPLATE,
    MOVIE_DIRECT_PREDICT_SYSTEM_PROMPT,
    MOVIE_DIRECT_PREDICT_USER_PROMPT
)
from tqdm import tqdm
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

def reward_profile(ground_truths, predicted):
    """
    ground_truths: ['like', 'neutral', 'like', ...]
    predicted:     ['neutral', 'neutral', 'like', ...]

    返回 reward（浮点数）
    """
    reward = 0.0

    for gt, pred in zip(ground_truths, predicted):
        if gt == pred:
            reward += 1.0
        # elif (
        #     (gt == "like" and pred == "dislike") or
        #     (gt == "dislike" and pred == "like")
        # ):
        #     reward -= 1.0
        else:
            reward += -1.0

    return reward

def rating_to_mood_fn(rating):
    """
    rating: 数值型评分，范围 1-5

    返回：'like' / 'neutral' / 'dislike'
    """

    if rating >= 4.0:
        return "like"
    elif rating <= 2.0:
        return "dislike"
    else:
        return "neutral"

def profile_controller_compute_reward(data_source, solution_str, ground_truth, extra_info=None):
    """
    pred_profile: 模型生成的用户画像文本
    next_session: 真实 session 列表，每项含 rating/title/genres
    api_server:   预测情绪的 LLM API 地址

    返回：float reward
    """

    # print("====custom reward function called====")
    # print(solution_str)
    # print("=====================================")
    next_session = ground_truth["next_session"]
    pred_profile = solution_str

    # ground-truth moods
    gt_moods = [rating_to_mood_fn(item["rating"]) for item in next_session]

    # predicted moods

    pred_moods = []
    for item in next_session:
        user_prompt=MOVIE_PROFILE_SENTIMENT_PREDICTION_USER_PROMPT.format(
            user_profile=pred_profile,
            item=f"Title: {item['title']}\nGenres: {item['genres']}"
        )
        pred_moods.append(call(MOVIE_PROFILE_SENTIMENT_PREDICTION_SYSTEM_PROMPT, user_prompt,seed=42))

    # 调用你的 reward 机制
    reward = reward_profile(gt_moods, pred_moods)

    return {
        "score": reward,
        "solution_str": solution_str,
        "gt_moods": gt_moods,
        "pred_moods": pred_moods,
        "user_id":extra_info["user_id"]
    }


def profile_generation_compute_reward(data_source, solution_str, ground_truth, extra_info=None):
    """
    pred_profile: 模型生成的用户画像文本
    next_session: 真实 session 列表，每项含 rating/title/genres
    api_server:   预测情绪的 LLM API 地址

    返回：float reward
    """

    # print("====custom reward function called====")
    # print(solution_str)
    # print("=====================================")
    # print(ground_truth)
    # print("=====================================")
    
    pred_profile = solution_str

    session_now=ground_truth["session_now"]
    next_session=ground_truth["next_session"]

    # print("session_now:", session_now)
    # print("next_session:", next_session)


    # ground-truth moods
    gt_moods = [rating_to_mood_fn(item["rating"]) for item in next_session]

    logs=[]
    for item in session_now:
        tags = item.get("tags") 
        tags_part = f", tags={','.join(tags)}" if tags!=[] else "" 
        log=MOVIE_SESSION_ITEM_TEMPLATE.format( 
            title=item["title"], 
            genres=item["genres"],
            rating=item["rating"], 
            tags_part=tags_part 
        ) 
        logs.append(log)
    pred_moods = []
    for item in next_session:
        user_prompt=MOVIE_SENTIMENT_PREDICTION_USER_PROMPT.format(
            user_history="\n".join(logs),
            user_profile=pred_profile,
            item=f"Title: {item['title']}\nGenres: {item['genres']}"
        )
        pred_moods.append(call(MOVIE_SENTIMENT_PREDICTION_SYSTEM_PROMPT, user_prompt))

    # 调用你的 reward 机制
    reward = reward_profile(gt_moods, pred_moods)

    return {
        "score": reward,
        "solution_str": solution_str,
        "gt_moods": gt_moods,
        "pred_moods": pred_moods,
        "user_id":extra_info["user_id"]
    }


def log_recommendation(user_logs_path, log_num):
    with open(user_logs_path, "r") as f:
        user_logs = json.load(f)

    correct = 0
    total = 0

    for logs in tqdm(user_logs.values(), desc="Evaluating users"):
        if len(logs) < log_num + 1:
            continue
        

        history_logs = logs[-log_num-1:-1]   # -1 all
        target_item = logs[-1]

        print("history length",len(history_logs))

        history_text = "\n".join(
            MOVIE_SESSION_ITEM_TEMPLATE.format(
                title=item.get("title", ""),
                genres=item.get("genres", ""),
                rating=item.get("rating", ""),
                tags=",".join(item.get("tags", [])) 
            )
            for item in history_logs
        )

        user_prompt = MOVIE_DIRECT_PREDICT_USER_PROMPT.format(
            history=history_text,
            item=f"Title: {target_item['title']}\nGenres: {target_item['genres']}"
        )

        pred = call(MOVIE_DIRECT_PREDICT_SYSTEM_PROMPT, user_prompt,seed=42)
        gt = rating_to_mood_fn(target_item["rating"])

        print(f"Predicted: {pred}, Ground Truth: {gt}")

        if pred == gt:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"Accuracy: {acc:.4f}, Correct: {correct}, Total: {total}")
    return acc, correct, total


if __name__=="__main__":
    user_logs_path="/home/aiscuser/yifeisun/datasets/ml-10M100K/top500_user_sample_16_user_logs.json"
    log_recommendation(user_logs_path, -1)
