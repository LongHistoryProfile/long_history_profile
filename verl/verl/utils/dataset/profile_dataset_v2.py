import json
from collections import defaultdict
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


from prompt_config import PROFILE_CONTROLLER_SYSTEM_PROMPT,PROFILE_CONTROLLER_USER_PROMPT
import torch
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
import numpy as np

def format_evidence_block(step_reviews):
    return "\n".join(
        f"Item: {r['item_title']}| Rating: {r['rating']}\nReview: {r['title']} {r['text'].strip()[:200]}"
        for r in step_reviews
    )
import os
import json
from typing import Dict


def load_user_profiles_from_rollout(
    rollout_data_dir: str,
    global_steps: int,
) -> Dict[str, str]:
    """
    Load cand[0]['user_profile'] from rollout json files of a given global_steps.

    Returns:
        dict: {user_id: user_profile}
    """
    user_profile_dict = {}

    prefix = f"{global_steps}_"

    if not os.path.isdir(rollout_data_dir):
        raise ValueError(f"rollout_data_dir not found: {rollout_data_dir}")

    none_cnt=0

    for fname in os.listdir(rollout_data_dir):
        # 只加载当前 step
        if not fname.startswith(prefix) or not fname.endswith(".json"):
            continue

        fpath = os.path.join(rollout_data_dir, fname)

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {fname}: {e}")
            continue

        user_id = data.get("user_id")
        candidates = data.get("candidates", [])

        if not user_id:
            print(f"[WARN] Missing user_id in {fname}")
            continue

        if not candidates or not isinstance(candidates, list):
            print(f"[WARN] Empty or invalid candidates in {fname}")
            continue

        user_profile = candidates[0].get("user_profile")

        if user_profile is None:
            none_cnt+=1
            # 合法情况：模型没产 profile
            continue

        user_profile_dict[user_id] = user_profile
    print(f"Total profiles loaded for global_steps {global_steps}: {len(user_profile_dict)}")
    print(f"Total none profiles for global_steps {global_steps}: {none_cnt}")

    return user_profile_dict

class MultiTurnUserHistoryDataset(Dataset):
    """
    Each sample = one user episode (full chronological history).
    Prompt construction will be done in tool/env.
    """

    def __init__(
        self,
        data_files=None,
        config=None,
        tokenizer=None,
        max_samples: int = -1,
        is_train: bool = True,
        processor=None,
    ):
        self.config = config
        self.is_train = is_train
        self.tokenizer = tokenizer

        if self.is_train:
            self.df = pd.read_pickle(self.config.get("train_path"))
        else:
            self.df = pd.read_pickle(self.config.get("test_path"))

        # self.df=self.df[:10000]  # For debugging purpose

        
        self.start_idx=self.config.get("start_idx")

        validation_data_dir = self.config.get("prev_validation_data_dir", None)
        global_steps = self.config.get("prev_global_steps", None)

        if validation_data_dir is not None and global_steps is not None:
            self.user_profiles = load_user_profiles_from_rollout(
                rollout_data_dir=validation_data_dir,
                global_steps=global_steps,
            )
        else:
            print("No previous rollout data provided, user profiles will not be used.")
            self.user_profiles = None

        self.samples = self._build_samples(max_samples=max_samples)



    def _build_samples(self, max_samples=-1):
        users = defaultdict(list)
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="[Dataset] Group user reviews"):
            users[row["user_id"]].append(
                {
                    "rating": float(row["rating"]),
                    "title": row["title"],
                    "text": row["text"],
                    "user_id": row["user_id"],
                    "item_id": row["item_id"],
                    "timestamp": int(row["timestamp"]),
                    "item_title": row.get("item_title", ""),
                    "description": row.get("description", ""),
                    "item_average_rating": float(row.get("item_average_rating", 0.0)),
                    "item_rating_number": int(row.get("item_rating_number", 0)),
                }
            )
        print(f"==============total user number when is train {self.is_train}")
        print(len(users))

        processed = []
        for uid, reviews in tqdm(users.items(), desc="[Dataset] Build user episodes"):
            reviews.sort(key=lambda x: x["timestamp"])
            if len(reviews)-1 <=self.start_idx:
                continue
            processed.append(
                {
                    "user_id": uid,
                    "reviews": reviews,
                    "start_idx": self.start_idx,
                }
            )
        # processed = sorted(processed, key=lambda x: x["user_id"])[:10]

        episode_step_size = self.config.get("episode_step_size", 1)
        max_prompt_length = self.config.get("max_prompt_length", 4096)
        truncation = self.config.get("truncation", "left")
        apply_chat_template_kwargs = self.config.get("apply_chat_template_kwargs", None)
        tokenizer = self.tokenizer


        samples=[{} for _ in range(len(processed))]
        
        for id, p in enumerate(processed):
            user_id = p["user_id"]
            reviews = p["reviews"]
            start_idx = p["start_idx"]

            samples[id]["reward_model"]={}
            samples[id]["reward_model"]["ground_truth"]=p
            
            end_idx = min(start_idx + episode_step_size, len(reviews) - 1)

            if start_idx>=end_idx:
                continue

            step_reviews = reviews[start_idx:end_idx]

            evidence_block = format_evidence_block(step_reviews)
            # existing_profile = prev_gt.get("current_profile", "No profile yet.")
            if self.user_profiles is not None and user_id in self.user_profiles:
                existing_profile = self.user_profiles[user_id]
            else:
                existing_profile = "No profile yet."
            samples[id]["reward_model"]["ground_truth"]["existing_profile"]= existing_profile
            

            user_prompt = PROFILE_CONTROLLER_USER_PROMPT.format(
                existing_profile=existing_profile,
                recent_reviews=evidence_block,
            )

            messages = [
                {"role": "system", "content": PROFILE_CONTROLLER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            raw_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs,
            )

            model_inputs = tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = model_inputs["input_ids"], model_inputs["attention_mask"]

            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_prompt_length,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=True,
                truncation=truncation,
            )

            position_ids = compute_position_id_with_mask(attention_mask)

            raw_prompt_ids = tokenizer.encode(raw_prompt, add_special_tokens=False)
            if len(raw_prompt_ids) > max_prompt_length:
                if truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]
                elif truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[:max_prompt_length]
                elif truncation == "middle":
                    left_half = max_prompt_length // 2
                    right_half = max_prompt_length - left_half
                    raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                elif truncation == "error":
                    raise RuntimeError(
                        f"Prompt length {len(raw_prompt_ids)} is longer than {max_prompt_length}."
                    )  
            samples[id]["input_ids"]=input_ids[0]
            samples[id]["attention_mask"]=attention_mask[0]
            samples[id]["position_ids"]=position_ids[0]
            samples[id]["raw_prompt_ids"]=raw_prompt_ids

            if self.config.get("return_full_prompt", True):
                samples[id]["full_prompts"]=raw_prompt
            if self.config.get("return_raw_chat", True):
                samples[id]["raw_prompt"]=messages

            
            samples[id]["reward_model"]["ground_truth"]["start_idx"]= end_idx

            start_idx = end_idx      

        print(f"==============total sample number when is train {self.is_train}")
        print(len(samples))     

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample=self.samples[idx]
        # row_dict={}
        # # row_dict["agent_name"]="profile_db_agent"
        # row_dict["reward_model"]={}
        # row_dict["reward_model"]["ground_truth"]=sample
        sample["data_source"]="lstm_profile"
        return sample

    @classmethod
    def build_profile_controller_batch_inputs(
        cls,
        batch_dict,
        tokenizer,
        max_prompt_length,
        truncation="left",
        episode_step_size=1,
        return_raw_chat=True,
        return_full_prompt=True,
        apply_chat_template_kwargs=None,
    ):
        """
        batch_dict: B 个 user episode
        本函数只构造 observation，不改变 start_idx
        evidence = reviews[start_idx : start_idx + episode_step_size]
        """



        if apply_chat_template_kwargs is None:
            apply_chat_template_kwargs = {}
        keys = list(batch_dict.keys())
        B = len(batch_dict[keys[0]])


        input_ids_list, attn_list, pos_list = [], [], []
        raw_prompt_ids_list = []
        raw_prompt_list, messages_list = [], []

        valid_indices = []

        for i in range(B):

            prev_gt = batch_dict["reward_model"][i]["ground_truth"]

            user_id = prev_gt["user_id"]
            reviews = prev_gt["reviews"]
            start_idx = int(prev_gt["start_idx"])

            end_idx = min(start_idx + episode_step_size, len(reviews) - 1)

            if start_idx>=end_idx:
                continue
            valid_indices.append(i)
            step_reviews = reviews[start_idx:end_idx]

            evidence_block = format_evidence_block(step_reviews)
            existing_profile = prev_gt.get("current_profile", "No profile yet.")

            user_prompt = PROFILE_CONTROLLER_USER_PROMPT.format(
                existing_profile=existing_profile,
                recent_reviews=evidence_block,
            )

            messages = [
                {"role": "system", "content": PROFILE_CONTROLLER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            raw_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs,
            )

            model_inputs = tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = model_inputs["input_ids"], model_inputs["attention_mask"]

            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_prompt_length,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=True,
                truncation=truncation,
            )

            position_ids = compute_position_id_with_mask(attention_mask)

            raw_prompt_ids = tokenizer.encode(raw_prompt, add_special_tokens=False)
            if len(raw_prompt_ids) > max_prompt_length:
                if truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]
                elif truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[:max_prompt_length]
                elif truncation == "middle":
                    left_half = max_prompt_length // 2
                    right_half = max_prompt_length - left_half
                    raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                elif truncation == "error":
                    raise RuntimeError(
                        f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                    )

            input_ids_list.append(input_ids[0])
            attn_list.append(attention_mask[0])
            pos_list.append(position_ids[0])
            raw_prompt_ids_list.append(raw_prompt_ids)

            if return_full_prompt:
                raw_prompt_list.append(raw_prompt)
            if return_raw_chat:
                messages_list.append(messages)
            batch_dict["reward_model"][i]["ground_truth"]["start_idx"]= end_idx
        
        if len(valid_indices) == 0:
            return None

        batch_dict["reward_model"] = np.array([batch_dict["reward_model"][i] for i in valid_indices], dtype=object)

        if "data_source" in batch_dict:
            batch_dict["data_source"] = np.array([batch_dict["data_source"][i] for i in valid_indices], dtype=object)

        row_dict = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_list),
            "position_ids": torch.stack(pos_list),
            "raw_prompt_ids":np.array(raw_prompt_ids_list, dtype=object),
        }

        if return_raw_chat:
            row_dict["raw_prompt"] = np.array(messages_list, dtype=object)
        if return_full_prompt:
            row_dict["full_prompts"] = np.array(raw_prompt_list, dtype=object)

        batch_dict.update(row_dict)

        return batch_dict
