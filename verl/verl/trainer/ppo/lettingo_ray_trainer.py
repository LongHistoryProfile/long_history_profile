from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import json
import os
from verl.utils.debug import marked_timer
from verl import DataProto
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # numpy scalar, e.g. np.float32
        return obj.item()
    elif torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    else:
        return obj


class LettingoRayPPOTrainer(RayPPOTrainer):
    def _log_rollout_data(
        self,
        batch: DataProto,
        reward_extra_infos_dict: dict,
        timing_raw: dict,
        rollout_data_dir: str,
        validate: bool = False,
    ):
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):

            os.makedirs(rollout_data_dir, exist_ok=True)

            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()

            reward_models = batch.non_tensor_batch["reward_model"]
            total = len(inputs)

            pred_ratings = reward_extra_infos_dict.get("predicted_rating", [None] * total)
            gt_ratings = reward_extra_infos_dict.get("gt_rating", [None] * total)
            
            reasons= reward_extra_infos_dict.get("reason", [None] * total)
            user_profiles= reward_extra_infos_dict.get("user_profile", [None] * total)
            groups = {}

            for i in range(total):
                gt = reward_models[i]["ground_truth"]
                user_id = str(gt["user_id"])
                start_idx = str(gt["start_idx"])
                existing_profile=gt.get("existing_profile", None)

                reason = reasons[i]
                user_profile = user_profiles[i]


                key = (user_id, start_idx)

                if key not in groups:
                    groups[key] = {
                        "step": self.global_steps,
                        "user_id": user_id,
                        "start_idx": start_idx,
                        "existing_profile":existing_profile,
                        "prompt": inputs[i],
                        "candidates": [],
                    }

                cand = {
                    "response": outputs[i],
                    "reason": reason,
                    "user_profile": user_profile,
                    "score": float(scores[i]),
                    "predicted_rating": None if pred_ratings[i] is None else float(pred_ratings[i]),
                    "gt_rating": None if gt_ratings[i] is None else float(gt_ratings[i]),
                }

                groups[key]["candidates"].append(cand)

            for (user_id, start_idx), entry in groups.items():
                entry = make_json_safe(entry)
                filename = os.path.join(
                    rollout_data_dir,
                    f"{self.global_steps}_{user_id}_{start_idx}.json"
                )
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False, indent=2))

            print(f"Dumped {len(groups)} groups to {rollout_data_dir}")