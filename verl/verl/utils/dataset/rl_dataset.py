# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
import traceback
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \\*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
        is_train: bool = True,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.image_patch_size = config.get("image_patch_size", 14)
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.tool_config_path = config.get("tool_config_path", None)
        self.tool_schemas = None
        if self.tool_config_path:
            try:
                from verl.tools.utils.tool_registry import initialize_tools_from_config

                tool_list = initialize_tools_from_config(self.tool_config_path)
                # match ToolAgentLoop behaviour: model_dump to plain dicts
                self.tool_schemas = [
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
                ]
            except Exception as e:
                logger.warning("Failed to initialize tools from %s: %s", self.tool_config_path, e)
                self.tool_schemas = None

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    try:
                        messages = self._build_messages(doc)
                        # pass tool schemas if available so the processor can format prompts
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        raw_prompt = self.processor.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
                        )
                        if image_key in doc and doc[image_key]:
                            images = [
                                process_image(image, image_patch_size=self.image_patch_size) for image in doc[image_key]
                            ]
                        else:
                            images = None

                        if video_key in doc and doc[video_key]:
                            videos, video_metadata = zip(
                                *[
                                    process_video(
                                        video, image_patch_size=self.image_patch_size, return_video_metadata=True
                                    )
                                    for video in doc[video_key]
                                ],
                                strict=True,
                            )
                            videos = list(videos)
                            video_metadata = list(video_metadata)
                            videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}
                        else:
                            videos = None
                            videos_kwargs = {}

                        return len(
                            processor(text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs)[
                                "input_ids"
                            ][0]
                        )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            else:

                def doc2len(doc) -> int:
                    try:
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        return len(
                            tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True, **apply_kwargs)
                        )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [process_image(image, image_patch_size=self.image_patch_size) for image in row_dict_images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            videos_kwargs = {}
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos, video_metadata = zip(
                    *[
                        process_video(video, image_patch_size=self.image_patch_size, return_video_metadata=True)
                        for video in row_dict_videos
                    ],
                    strict=True,
                )
                videos = list(videos)
                video_metadata = list(video_metadata)
                videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [
                    (video.numpy(), metadata) for video, metadata in zip(videos, video_metadata, strict=True)
                ]

            model_inputs = self.processor(
                text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs, return_tensors="pt"
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs

        # print(row_dict)


        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()



from prompt_config import (
    MOVIE_SESSION_ITEM_TEMPLATE,
    MOVIE_MEMORY_CONTROLLER_USER_PROMPT,
    MOVIE_MEMORY_CONTROLLER_SYSTEM_PROMPT,
    MOVIE_PROFILE_GENERATION_SYSTEM_PROMPT,
    MOVIE_PROFILE_GENERATION_USER_PROMPT,
    MOVIE_SESSION_ITEM_TEMPLATE_NO_TAG,
    DUET_PROFILE_SYSTEM_PROMPT,
    DUET_PROFILE_GENERATOR_PROMPT,
    LettinGo_PROFILE_SYSTEM_PROMPT,
    LettinGo_PROFILE_GENERATOR_PROMPT
)

import json

import pickle
import pandas as pd

from tqdm import tqdm

def yelp_adaptive(df):
    df["unixReviewTime"]=(
    pd.to_datetime(df["date"], utc=True)
    .astype("int64") // 10**9
    )    

    return df
    
class LettinGoDataset(Dataset):
    """
    A dataset class for generating textual profiles for DUET framework based on the user's data and items.
    """

    def __init__(        
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[object] = None,
        max_samples: int = -1,
        is_train=True):
        """
        Initialize the dataset with the .pkl file and configuration settings.
        
        :param data_file: Path to the .pkl file containing user-item interaction data.
        :param tokenizer: Pre-trained tokenizer for tokenizing input prompts.
        :param config: Configuration for dataset handling.
        """
        self.config=config
        self.is_train=is_train


        # 路径相关配置
        if self.is_train:
            self.train_df=pd.read_pickle(self.config.get("train_path"))
            if "Yelp" in self.config.get("train_path"):
                self.train_df=yelp_adaptive(self.train_df)
        else:
            self.train_df=pd.read_pickle(self.config.get("train_path"))
            self.test_df=pd.read_pickle(self.config.get("test_path"))
            if "Yelp" in self.config.get("train_path"):
                self.train_df=yelp_adaptive(self.train_df)  
                self.test_df=yelp_adaptive(self.test_df)

            
        # prompt 长度控制相关
        self.max_prompt_length = int(config.get("max_prompt_length", 1024))
        self.truncation = config.get("truncation", "error")  # 'left' / 'right' / 'middle' / 'error'

        # 其它选项
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed", None)

        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.tokenizer = tokenizer
        self.config = config
        self._build_samples()


    def _build_samples(self):
        if self.is_train:
            df = self.train_df
        else:
            df = self.test_df

        # df=df.iloc[:1000]  # for debug
        processed = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="[LettinGoDataset] Building samples"):
    
            uid, iid = row['user_id'], row['item_id']
            user_title = row.get('reviewerName', f"User_{uid}")
            item_title = row.get('title', f"Item_{iid}")
            gt_rating = float(row['ratings'])
            current_time = row['unixReviewTime']  # 关键：当前样本的时间戳，用于过滤历史
            
            # 1. 严格过滤用户历史（仅保留当前时间之前的记录）

            user_history = df[
                (df['user_id'] == uid) & 
                (df['unixReviewTime'] < current_time)  # 仅历史数据
            ]
            if self.is_train is False:
                train_df=self.train_df
                user_history = pd.concat(
                [
                    user_history,
                    train_df[(train_df['user_id'] == uid)& (train_df['unixReviewTime'] < current_time)]
                    
                ] )
            user_history = user_history.sort_values('unixReviewTime').tail(10)  # 最近10条
            
            # 2. 严格过滤物品历史（仅保留当前时间之前的记录）
            item_history = df[
                (df['item_id'] == iid) & 
                (df['unixReviewTime'] < current_time)  # 仅历史数据
            ]
            if self.is_train is False:
                train_df=self.train_df
                item_history = pd.concat(
                [
                    item_history,
                    train_df[(train_df['item_id'] == iid)& (train_df['unixReviewTime'] < current_time)]
                    
                ] )
            
            item_history = item_history.sort_values('unixReviewTime').tail(10)  # 最近10条

            if self.is_train:
                if len(user_history) < 1:
                    continue  # 训练时严格要求用户有历史数据，但物品的title可以提供信息
            
            # 3. 处理历史数据缺失的情况
            if user_history.empty:
                user_avg_rating = "N/A (no historical data)"
                user_history_text = "[No historical interactions available for this user]"
            else:
                user_avg_rating = f"{user_history['ratings'].mean():.1f}"
                user_history_text = "\n".join([
                    f"[History {i+1}] Item: {r['title']}, Rating: {r['ratings']:.1f}\nReview: {r.get('reviews', '').strip()[:200]}"
                    for i, (_, r) in enumerate(user_history.iterrows())
                ])
            
            if item_history.empty:
                item_avg_rating = "N/A (no historical data)"
                item_history_text = "[No historical reviews available for this item]"
            else:
                item_avg_rating = f"{item_history['ratings'].mean():.1f}"
                item_history_text = "\n".join([
                    f"[Review {i+1}] User: {r.get('reviewerName', 'Anonymous')}, Rating: {r['ratings']:.1f}\nReview: {r.get('reviews', '').strip()[:200]}"
                    for i, (_, r) in enumerate(item_history.iterrows())
                ])
            
            profile_prompt = LettinGo_PROFILE_GENERATOR_PROMPT.format(
                user_title=user_title,
                item_title=item_title,
                user_history_text=user_history_text,
                item_history_text=item_history_text,
                user_avg_rating=user_avg_rating,
                item_avg_rating=item_avg_rating
            )
            
            processed.append({
                "messages": [
                    {"role": "system", "content": LettinGo_PROFILE_SYSTEM_PROMPT},
                    {"role": "user", "content": profile_prompt}
                ],
                "gt_rating": gt_rating,
                "user_title": user_title,
                "item_title": item_title,
                "user_id": uid,
                "item_id": iid,
                "user_avg_rating": user_avg_rating,
                "item_avg_rating": item_avg_rating
            })
        self.samples = processed
        print(f"[LettinGoDataset] Total samples built: {len(self.samples)}")    


    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.
        
        :param idx: Index of the sample to retrieve.
        :return: The sample's input features for the model.
        """
        row_dict: dict = {}
        sample = self.samples[idx]
        messages = sample["messages"]
        model_inputs = {}
        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        try:
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
        except:
            print("Error in postprocess_data for sample index:", idx, "with  length:", len(input_ids[0]))
            print("===============messages=================")
            print(messages)

            


        # 3) position_ids
        position_ids = compute_position_id_with_mask(attention_mask)
        # 核心输入张量
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        # 4) raw_prompt_ids（不用于输入，只是对齐 RLHFDataset 的接口）
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )


        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt   

        # 辅助信息（方便 reward function 用）
        # row_dict["user_id"] = sample["user_id"]
        # row_dict["sess_idx"] = sample["sess_idx"]
        # row_dict["base_profile"] = sample["base_profile"]
        # row_dict["next_session"] = sample["next_session"]

        row_dict["data_source"] ="DUET"
        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()

        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["extra_info"]["messages"] = messages
        # tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        # interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        # need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        # if need_tools_kwargs and not tools_kwargs:
        #     logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        # row_dict["tools_kwargs"] = tools_kwargs
        # row_dict["interaction_kwargs"] = interaction_kwargs
        row_dict["reward_model"] = {}
        row_dict["reward_model"]["ground_truth"] = {
            "user_title": sample["user_title"],
            "item_title": sample["item_title"],
            "user_avg_rating": sample["user_avg_rating"],
            "item_avg_rating": sample["item_avg_rating"],
            "gt_rating": sample["gt_rating"]
        }


        return row_dict
    

    def __len__(self):
        return len(self.samples)

from prompt_config import PROFILE_CONTROLLER_SYSTEM_PROMPT, PROFILE_CONTROLLER_USER_PROMPT
# def np_obj(x):
#     """Pack arbitrary python object into numpy object array (0-d)."""
#     return np.array(x, dtype=object)

class MultiTurnUserHistoryDataset(Dataset):
    """
    Each sample = one user episode (full chronological history).
    Prompt construction will be done in tool/env.
    """

    def __init__(
        self,
        config,
        tokenizer=None,
        max_samples: int = -1,
        is_train: bool = True,
    ):
        self.config = config
        self.is_train = is_train
        self.tokenizer = tokenizer

        if self.is_train:
            self.df = pd.read_pickle(self.config.get("train_path"))
        else:
            self.df = pd.read_pickle(self.config.get("test_path"))

        # load summary dict (optional, can be used for initial profile seed)
        with open(self.summary_path, "r") as f:
            self.summary_dict = json.load(f)

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

        processed = []
        for uid, reviews in tqdm(users.items(), desc="[Dataset] Build user episodes"):
            reviews.sort(key=lambda x: x["timestamp"])

            processed.append(
                {
                    "user_id": uid,
                    "reviews": reviews,
                }
            )

        if max_samples > 0:
            processed = processed[:max_samples]
        return processed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
class MultiTurnProfileControllerDataset(Dataset):
    """
    Multi-turn sliding-window dataset for profile controller RL training.

    Each sample = one user episode:
      windows: [0:5], [5:10], ...
      targets: review[5], review[10], ...
    """

    def __init__(
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[object] = None,
        max_samples: int = -1,
        is_train: bool = True,
    ):
        self.config = config
        self.is_train = is_train
        self.tokenizer = tokenizer

        if self.is_train:
            self.df = pd.read_pickle(self.config.get("train_path"))
            self.summary_path = self.config.get("train_summary_path")
        else:
            self.df = pd.read_pickle(self.config.get("test_path"))
            self.summary_path = self.config.get("test_summary_path")

        # self.df = self.df[:1] # for
        # prompt length
        self.max_prompt_length = int(config.get("max_prompt_length", 1024))
        self.truncation = config.get("truncation", "error")

        # flags
        self.return_raw_chat = bool(config.get("return_raw_chat", False))
        self.return_full_prompt = bool(config.get("return_full_prompt", False))
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})


        self._build_samples(max_samples=max_samples)

    def _build_samples(self, max_samples=-1):
        with open(self.summary_path, "r") as f:
            summary_dict = json.load(f)

        # group by user_id
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
                    "item_title": row["item_title"],
                    "description": row["description"],
                    "item_average_rating": float(row["item_average_rating"]),
                    "item_rating_number": int(row["item_rating_number"]),
                }
            )

        processed = []

        for uid in tqdm(users, desc="[Dataset] Build multi-turn episodes"):
            reviews = users[uid]
            reviews.sort(key=lambda x: x["timestamp"])

            init_profile = summary_dict[uid].get("merged_summary", "")
            if not init_profile:
                continue

            # build windows+targets
            windows, targets = [], []
            t = 0
            while True:
                start = t * self.stride
                end = start + self.window_size
                target_idx = end
                if target_idx >= len(reviews):
                    break
                window = reviews[start:end]
                if len(window) < self.window_size:
                    break
                windows.append(window)
                targets.append(reviews[target_idx])
                t += 1

            if len(windows) == 0:
                continue

            # safety cap
            if self.max_turns > 0:
                windows = windows[: self.max_turns]
                targets = targets[: self.max_turns]

            # system-only; tool.create injects first user prompt
            messages = [{"role": "system", "content": PROFILE_CONTROLLER_SYSTEM_PROMPT}]
            raw_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]

            try:
                input_ids, attention_mask = verl_F.postprocess_data(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.truncation,
                )
            except Exception:
                continue

            processed.append(
                {
                    "user_id": uid,
                    "messages": messages,
                    "raw_prompt": raw_prompt,
                    "input_ids": input_ids[0],
                    "attention_mask": attention_mask[0],
                    "init_profile": init_profile,
                    "windows": windows,
                    "targets": targets,
                }
            )

            if max_samples > 0 and len(processed) >= max_samples:
                break

        # print("==========================debug: one example ==========================")
        # if len(processed) > 0:
        #     print (processed[0])

        self.samples = processed
        print(f"[Dataset] Total episodes: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        row_dict = {}

        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = compute_position_id_with_mask(attention_mask)

        # raw_prompt_ids
        raw_prompt_ids = self.tokenizer.encode(sample["raw_prompt"], add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            else:
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} > {self.max_prompt_length}")
        row_dict["raw_prompt_ids"] = raw_prompt_ids

        # Optional debug
        if self.return_raw_chat:
            row_dict["raw_prompt"] = sample["messages"]
        if self.return_full_prompt:
            row_dict["full_prompts"] = sample["raw_prompt"]

        row_dict["data_source"] = "ProfileControllerMultiTurn"

        row_dict["tools_kwargs"] = {
                "user_id": sample["user_id"],
                "init_profile": sample["init_profile"],
                "windows": sample["windows"],
                "targets": sample["targets"], 
            }

        # # Ground truth for reward_fn (targets list)
        # row_dict["reward_model"] = {
        #         "ground_truth": {
        #             "targets": sample["targets"],
        #         }
        #     }

        # extra_info for debugging / fallback parsing
        row_dict["extra_info"] = {
                "messages": sample["messages"],
                "user_id": sample["user_id"],
            }

        row_dict["agent_name"]="tool_agent"

        return row_dict

class ProfileController(Dataset):
    """
    A dataset class for generating textual profiles for DUET framework based on the user's data and items.
    """

    def __init__(        
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[object] = None,
        max_samples: int = -1,
        is_train=True):
        """
        Initialize the dataset with the .pkl file and configuration settings.
        
        :param data_file: Path to the .pkl file containing user-item interaction data.
        :param tokenizer: Pre-trained tokenizer for tokenizing input prompts.
        :param config: Configuration for dataset handling.
        """
        self.config=config
        self.is_train=is_train
        # 路径相关配置
        if self.is_train:
            self.train_df=pd.read_pickle(self.config.get("train_path")) 
            self.train_summary_path=self.config.get("train_summary_path")
        else:
            self.test_df=pd.read_pickle(self.config.get("test_path"))
            self.test_summary_path=self.config.get("test_summary_path")

            
        # prompt 长度控制相关
        self.max_prompt_length = int(config.get("max_prompt_length", 1024))
        self.truncation = config.get("truncation", "error")  # 'left' / 'right' / 'middle' / 'error'

        # 其它选项
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed", None)

        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.tokenizer = tokenizer
        self.config = config
        self._build_samples()


    def _build_samples(self):
        if self.is_train:
            df = self.train_df
            summary_path=self.train_summary_path
        else:
            df = self.test_df
            summary_path=self.test_summary_path

        # df=df.iloc[:1000]  # for debug
        processed = []
        summary_dict={}
        with open(summary_path, "r") as f:
            summary_dict = json.load(f)

        
        users = defaultdict(list)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="[ProfileController] Building user reviews"):
            users[row["user_id"]].append({
                "rating": float(row["rating"]),
                "title": row["title"],
                "text": row["text"],
                "user_id": row["user_id"],
                "item_id": row["item_id"],
                "timestamp": int(row["timestamp"]),
                "item_title": row["item_title"],
                "description": row["description"],
                "item_average_rating": float(row["item_average_rating"]),
                "item_rating_number": int(row["item_rating_number"]),
            })

        #统计最长 最短 平均
        lengths = [len(users[uid]) for uid in users]
        print(f"Review counts - Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.2f}")

        long_ratio=self.config.get("long_ratio",0.9)
        print(f"Summary Profile using {long_ratio*100}% of reviews per user.")
        print("Truncating reviews...")
        # user 内按时间排序（必须）
        for uid in users:
            users[uid].sort(key=lambda x: x["timestamp"])
            # 只取前 90%的之后部分
            cutoff = int(len(users[uid]) * long_ratio)
            users[uid] = users[uid][cutoff:]

        #取后统计最长 最短 平均
        lengths = [len(users[uid]) for uid in users]
        print(f"Post-truncation review counts - Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.2f}")

        for uid in tqdm(users, desc="[ProfileController] Building samples from user reviews"):
            user_reviews = users[uid]
            taregt_review= user_reviews[5]
            user_reviews = user_reviews[:5]  

            try:
                user_profile_summary = summary_dict.get(uid)["merged_summary"]
            except KeyError:
                print(f"User ID {uid} not found in summary_dict. Skipping this user.")
                continue
            def build_review_sentence(r):
                return (
                    f"Item: {r['item_title']} "
                    f"Rating: {r['rating']} (item_avg_rating={r['item_average_rating']}, item_rating_number={r['item_rating_number']})\n"
                    f"Review: {r['title']}: {r['text'][:200].strip()}"
                )
            review_sentences = [build_review_sentence(r) for r in user_reviews]

            recent_reviews = "\n".join([
                    # f"[History {i+1}] Item: {r['title']}, Rating: {r['ratings']:.1f}\nReview: {r.get('reviews', '').strip()[:200]}"
                    # for i, (_, r) in enumerate(user_history.iterrows())
                    f"[Review {i+1}] {review_sentences[i]}"
                    for i in range(len(user_reviews))
                ])

            profile_prompt = PROFILE_CONTROLLER_USER_PROMPT.format(
                existing_profile=user_profile_summary,
                recent_reviews=recent_reviews
            )

            processed.append({
                "messages": [
                    {"role": "system", "content": PROFILE_CONTROLLER_SYSTEM_PROMPT},
                    {"role": "user", "content": profile_prompt}
                ],
                "user_id": uid,
                "target_review": taregt_review,
            })

        #统一tokenize
        max_len=0
        for pro in tqdm(processed, desc="[ProfileController] Pre-tokenizing samples"):
            raw_prompt = self.tokenizer.apply_chat_template(
                pro["messages"], add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
            try:
                input_ids, attention_mask = verl_F.postprocess_data(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.truncation,
                )
            except:
                print("Error in postprocess_data during pre-tokenization for user_id:", pro["user_id"], "with  length:", len(input_ids[0]))
                print("===============messages=================")
                print(pro["messages"][0]["content"])
                print("=======================================")
                print(pro["messages"][1]["content"])
            
            max_len=max(max_len, len(input_ids[0]))

            pro["raw_prompt"]= raw_prompt
            pro["input_ids"] = input_ids[0]
            pro["attention_mask"] = attention_mask[0]
        print(f"[ProfileController] Max tokenized prompt length: {max_len}")

        print("===========================message example============================")
        print(processed[0]["messages"][0]["content"])
        print("---------------------------------------------------------------------")
        print(processed[0]["messages"][1]["content"])
        print("=====================================================================")

        self.samples = processed
        print(f"[ProfileController] Total samples built: {len(self.samples)}")

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.
        
        :param idx: Index of the sample to retrieve.
        :return: The sample's input features for the model.
        """
        row_dict: dict = {}
        sample = self.samples[idx]
        messages = sample["messages"]
        # model_inputs = {}
        # raw_prompt = self.tokenizer.apply_chat_template(
        #     messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        # )
        # model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)

        # input_ids = model_inputs["input_ids"]
        # attention_mask = model_inputs["attention_mask"]

        # try:
        #     input_ids, attention_mask = verl_F.postprocess_data(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         max_length=self.max_prompt_length,
        #         pad_token_id=self.tokenizer.pad_token_id,
        #         left_pad=True,
        #         truncation=self.truncation,
        #     )
        # except:
        #     print("Error in postprocess_data for sample index:", idx, "with  length:", len(input_ids[0]))
        #     print("===============messages=================")
        #     print(messages)

        raw_prompt = sample["raw_prompt"]
        attention_mask = sample["attention_mask"]
        input_ids = sample["input_ids"]


        # 3) position_ids
        position_ids = compute_position_id_with_mask(attention_mask)
        # 核心输入张量
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        # 4) raw_prompt_ids（不用于输入，只是对齐 RLHFDataset 的接口）
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )


        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings


        # 辅助信息（方便 reward function 用）
        # row_dict["user_id"] = sample["user_id"]
        # row_dict["sess_idx"] = sample["sess_idx"]
        # row_dict["base_profile"] = sample["base_profile"]
        # row_dict["next_session"] = sample["next_session"]

        row_dict["data_source"] ="ProfileController"
        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()

        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["extra_info"]["messages"] = messages
        # tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        # interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        # need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        # if need_tools_kwargs and not tools_kwargs:
        #     logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        # row_dict["tools_kwargs"] = tools_kwargs
        # row_dict["interaction_kwargs"] = interaction_kwargs
        row_dict["reward_model"] = {}
        row_dict["reward_model"]["ground_truth"] = {
            "target_review": sample["target_review"]
        }
        return row_dict
    
    def __len__(self):
        return len(self.samples)


class DUETDataset(Dataset):
    """
    A dataset class for generating textual profiles for DUET framework based on the user's data and items.
    """

    def __init__(        
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[object] = None,
        max_samples: int = -1,
        is_train=True):
        """
        Initialize the dataset with the .pkl file and configuration settings.
        
        :param data_file: Path to the .pkl file containing user-item interaction data.
        :param tokenizer: Pre-trained tokenizer for tokenizing input prompts.
        :param config: Configuration for dataset handling.
        """
        self.config=config
        self.is_train=is_train
        # 路径相关配置
        if self.is_train:
            self.train_df=pd.read_pickle(self.config.get("train_path"))
            if "Yelp" in self.config.get("train_path"):
                self.train_df=yelp_adaptive(self.train_df)      
        else:
            self.train_df=pd.read_pickle(self.config.get("train_path"))
            self.test_df=pd.read_pickle(self.config.get("test_path"))
            if "Yelp" in self.config.get("train_path"):
                self.train_df=yelp_adaptive(self.train_df)  
                self.test_df=yelp_adaptive(self.test_df)

            
        # prompt 长度控制相关
        self.max_prompt_length = int(config.get("max_prompt_length", 1024))
        self.truncation = config.get("truncation", "error")  # 'left' / 'right' / 'middle' / 'error'

        # 其它选项
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed", None)

        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.tokenizer = tokenizer
        self.config = config
        self._build_samples()


    def _build_samples(self):
        if self.is_train:
            df = self.train_df
        else:
            df = self.test_df

        # df=df.iloc[:1000]  # for debug
        processed = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="[DuetDataset] Building samples"):
    
            uid, iid = row['user_id'], row['item_id']
            user_title = row.get('reviewerName',f"User_{uid}")
            # if not isinstance(user_title, str) or not user_title.strip():
            #     user_title = f"User_{uid}"

            item_title = row.get('title',f"Item_{iid}")
            # if not isinstance(item_title, str) or not item_title.strip():
            #     item_title = f"Item_{iid}"

            gt_rating = float(row['ratings'])
            current_time = row['unixReviewTime']  # 关键：当前样本的时间戳，用于过滤历史
            
            # 1. 严格过滤用户历史（仅保留当前时间之前的记录）

            user_history = df[
                (df['user_id'] == uid) & 
                (df['unixReviewTime'] < current_time)  # 仅历史数据
            ]
            if self.is_train is False:
                train_df=self.train_df
                user_history = pd.concat(
                [
                    user_history,
                    train_df[(train_df['user_id'] == uid)& (train_df['unixReviewTime'] < current_time)]
                    
                ] )
            user_history = user_history.sort_values('unixReviewTime').tail(10)  # 最近10条
            
            # 2. 严格过滤物品历史（仅保留当前时间之前的记录）
            item_history = df[
                (df['item_id'] == iid) & 
                (df['unixReviewTime'] < current_time)  # 仅历史数据
            ]
            if self.is_train is False:
                train_df=self.train_df
                item_history = pd.concat(
                [
                    item_history,
                    train_df[(train_df['item_id'] == iid)& (train_df['unixReviewTime'] < current_time)]
                    
                ] )
            
            item_history = item_history.sort_values('unixReviewTime').tail(10)  # 最近10条

            if self.is_train:
                if len(user_history) < 1:
                    continue  # 训练时严格要求用户有历史数据，但物品的title可以提供信息
            
            # 3. 处理历史数据缺失的情况
            if user_history.empty:
                user_avg_rating = "N/A (no historical data)"
                user_history_text = "[No historical interactions available for this user]"
            else:
                user_avg_rating = f"{user_history['ratings'].mean():.1f}"
                user_history_text = "\n".join([
                    f"[History {i+1}] Item: {r['title']}, Rating: {r['ratings']:.1f}\nReview: {r.get('reviews', '').strip()[:200]}"
                    for i, (_, r) in enumerate(user_history.iterrows())
                ])
            
            if item_history.empty:
                item_avg_rating = "N/A (no historical data)"
                item_history_text = "[No historical reviews available for this item]"
            else:
                item_avg_rating = f"{item_history['ratings'].mean():.1f}"
                item_history_text = "\n".join([
                    f"[Review {i+1}] User: {r.get('reviewerName', 'Anonymous')}, Rating: {r['ratings']:.1f}\nReview: {r.get('reviews', '').strip()[:200]}"
                    for i, (_, r) in enumerate(item_history.iterrows())
                ])
            
            profile_prompt = DUET_PROFILE_GENERATOR_PROMPT.format(
                user_title=user_title,
                item_title=item_title,
                user_history_text=user_history_text,
                item_history_text=item_history_text,
                user_avg_rating=user_avg_rating,
                item_avg_rating=item_avg_rating
            )
            
            processed.append({
                "messages": [
                    {"role": "system", "content": DUET_PROFILE_SYSTEM_PROMPT},
                    {"role": "user", "content": profile_prompt}
                ],
                "gt_rating": gt_rating,
                "user_title": user_title,
                "item_title": item_title,
                "user_id": uid,
                "item_id": iid,
                "user_avg_rating": user_avg_rating,
                "item_avg_rating": item_avg_rating
            })
        self.samples = processed

        if self.is_train:
            self.samples = processed[:10000]  # 仅使用前100k条进行训练，防止过大

        print(f"[DuetDataset] Total samples built: {len(self.samples)}")    


    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.
        
        :param idx: Index of the sample to retrieve.
        :return: The sample's input features for the model.
        """
        row_dict: dict = {}
        sample = self.samples[idx]
        messages = sample["messages"]
        model_inputs = {}
        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        try:
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
        except:
            print("Error in postprocess_data for sample index:", idx, "with  length:", len(input_ids[0]))
            print("===============messages=================")
            print(messages)

            


        # 3) position_ids
        position_ids = compute_position_id_with_mask(attention_mask)
        # 核心输入张量
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        # 4) raw_prompt_ids（不用于输入，只是对齐 RLHFDataset 的接口）
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )


        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings


        # 辅助信息（方便 reward function 用）
        # row_dict["user_id"] = sample["user_id"]
        # row_dict["sess_idx"] = sample["sess_idx"]
        # row_dict["base_profile"] = sample["base_profile"]
        # row_dict["next_session"] = sample["next_session"]

        row_dict["data_source"] ="DUET"
        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()

        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["extra_info"]["messages"] = messages
        # tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        # interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        # need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        # if need_tools_kwargs and not tools_kwargs:
        #     logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        # row_dict["tools_kwargs"] = tools_kwargs
        # row_dict["interaction_kwargs"] = interaction_kwargs
        row_dict["reward_model"] = {}
        row_dict["reward_model"]["ground_truth"] = {
            "user_title": sample["user_title"],
            "item_title": sample["item_title"],
            "user_avg_rating": sample["user_avg_rating"],
            "item_avg_rating": sample["item_avg_rating"],
            "gt_rating": sample["gt_rating"]
        }
        return row_dict
    
    def __len__(self):
        return len(self.samples)

class MusicDUETDataset(Dataset):
    """
    A dataset class for generating textual profiles for DUET framework based on the user's data and items.
    """

    def __init__(        
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[object] = None,
        max_samples: int = -1,
        is_train=True):
        """
        Initialize the dataset with the .pkl file and configuration settings.
        
        :param data_file: Path to the .pkl file containing user-item interaction data.
        :param tokenizer: Pre-trained tokenizer for tokenizing input prompts.
        :param config: Configuration for dataset handling.
        """
        self.config=config
        self.is_train=is_train
        # 路径相关配置
        if self.is_train:
            self.train_df=pd.read_pickle(self.config.get("train_path"))
        else:
            self.train_df=pd.read_pickle(self.config.get("train_path"))
            self.test_df=pd.read_pickle(self.config.get("test_path"))

            
        # prompt 长度控制相关
        self.max_prompt_length = int(config.get("max_prompt_length", 1024))
        self.truncation = config.get("truncation", "error")  # 'left' / 'right' / 'middle' / 'error'

        # 其它选项
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed", None)

        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.tokenizer = tokenizer
        self.config = config
        self._build_samples()


    def _build_samples(self):
        if self.is_train:
            df = self.train_df
        else:
            df = self.test_df

        # df=df.iloc[:1000]  # for debug
        processed = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="[MusicDuetDataset] Building samples"):
    
            uid, iid = row['user_id'], row['item_id']
            user_title = row.get('reviewerName', f"User_{uid}")
            item_title = row.get('title', f"Item_{iid}")
            gt_rating = float(row['ratings'])
            current_time = row['unixReviewTime']  # 关键：当前样本的时间戳，用于过滤历史
            
            # 1. 严格过滤用户历史（仅保留当前时间之前的记录）

            user_history = df[
                (df['user_id'] == uid) & 
                (df['unixReviewTime'] < current_time)  # 仅历史数据
            ]
            if self.is_train is False:
                train_df=self.train_df
                user_history = pd.concat(
                [
                    user_history,
                    train_df[(train_df['user_id'] == uid)& (train_df['unixReviewTime'] < current_time)]
                    
                ] )
            user_history = user_history.sort_values('unixReviewTime').tail(10)  # 最近10条
            
            # 2. 严格过滤物品历史（仅保留当前时间之前的记录）
            item_history = df[
                (df['item_id'] == iid) & 
                (df['unixReviewTime'] < current_time)  # 仅历史数据
            ]
            if self.is_train is False:
                train_df=self.train_df
                item_history = pd.concat(
                [
                    item_history,
                    train_df[(train_df['item_id'] == iid)& (train_df['unixReviewTime'] < current_time)]
                    
                ] )
            
            item_history = item_history.sort_values('unixReviewTime').tail(10)  # 最近10条

            if self.is_train:
                if len(user_history) < 1:
                    continue  # 训练时严格要求用户有历史数据，但物品的title可以提供信息
            
            # 3. 处理历史数据缺失的情况
            if user_history.empty:
                user_avg_rating = "N/A (no historical data)"
                user_history_text = "[No historical interactions available for this user]"
            else:
                user_avg_rating = f"{user_history['ratings'].mean():.1f}"
                user_history_text = "\n".join([
                    f"[History {i+1}] Item: {r['title']}, Rating: {r['ratings']:.1f}\nReview: {r.get('reviews', '').strip()[:200]}"
                    for i, (_, r) in enumerate(user_history.iterrows())
                ])
            
            if item_history.empty:
                item_avg_rating = "N/A (no historical data)"
                item_history_text = "[No historical reviews available for this item]"
            else:
                item_avg_rating = f"{item_history['ratings'].mean():.1f}"
                item_history_text = "\n".join([
                    f"[Review {i+1}] User: {r.get('reviewerName', 'Anonymous')}, Rating: {r['ratings']:.1f}\nReview: {r.get('reviews', '').strip()[:200]}"
                    for i, (_, r) in enumerate(item_history.iterrows())
                ])
            
            profile_prompt = DUET_PROFILE_GENERATOR_PROMPT.format(
                user_title=user_title,
                item_title=item_title,
                user_history_text=user_history_text,
                item_history_text=item_history_text,
                user_avg_rating=user_avg_rating,
                item_avg_rating=item_avg_rating
            )
            
            processed.append({
                "messages": [
                    {"role": "system", "content": DUET_PROFILE_SYSTEM_PROMPT},
                    {"role": "user", "content": profile_prompt}
                ],
                "gt_rating": gt_rating,
                "user_title": user_title,
                "item_title": item_title,
                "user_id": uid,
                "item_id": iid,
                "user_avg_rating": user_avg_rating,
                "item_avg_rating": item_avg_rating
            })
        self.samples = processed
        print(f"[DuetDataset] Total samples built: {len(self.samples)}")    


    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.
        
        :param idx: Index of the sample to retrieve.
        :return: The sample's input features for the model.
        """
        row_dict: dict = {}
        sample = self.samples[idx]
        messages = sample["messages"]
        model_inputs = {}
        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        try:
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
        except:
            print("Error in postprocess_data for sample index:", idx, "with  length:", len(input_ids[0]))
            print("===============messages=================")
            print(messages)

            


        # 3) position_ids
        position_ids = compute_position_id_with_mask(attention_mask)
        # 核心输入张量
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        # 4) raw_prompt_ids（不用于输入，只是对齐 RLHFDataset 的接口）
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )


        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings


        # 辅助信息（方便 reward function 用）
        # row_dict["user_id"] = sample["user_id"]
        # row_dict["sess_idx"] = sample["sess_idx"]
        # row_dict["base_profile"] = sample["base_profile"]
        # row_dict["next_session"] = sample["next_session"]

        row_dict["data_source"] ="DUET"
        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()

        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["extra_info"]["messages"] = messages
        # tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        # interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        # need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        # if need_tools_kwargs and not tools_kwargs:
        #     logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        # row_dict["tools_kwargs"] = tools_kwargs
        # row_dict["interaction_kwargs"] = interaction_kwargs
        row_dict["reward_model"] = {}
        row_dict["reward_model"]["ground_truth"] = {
            "user_title": sample["user_title"],
            "item_title": sample["item_title"],
            "user_avg_rating": sample["user_avg_rating"],
            "item_avg_rating": sample["item_avg_rating"],
            "gt_rating": sample["gt_rating"]
        }
        return row_dict
    


        

    def __len__(self):
        return len(self.samples)

class YelpDUETDataset(Dataset):
    """
    A dataset class for generating textual profiles for DUET framework based on the user's data and items.
    """

    def __init__(        
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[object] = None,
        max_samples: int = -1,
        is_train=True):
        """
        Initialize the dataset with the .pkl file and configuration settings.
        
        :param data_file: Path to the .pkl file containing user-item interaction data.
        :param tokenizer: Pre-trained tokenizer for tokenizing input prompts.
        :param config: Configuration for dataset handling.
        """
        self.config=config
        self.is_train=is_train
        # 路径相关配置
        def yelp_adaptive(df):
            df["unixReviewTime"]=(
            pd.to_datetime(df["date"], utc=True)
            .astype("int64") // 10**9
            )    

            return df
        if self.is_train:
            self.train_df=pd.read_pickle(self.config.get("train_path"))
            self.train_df=yelp_adaptive(self.train_df)
        else:
            self.train_df=pd.read_pickle(self.config.get("train_path"))
            self.test_df=pd.read_pickle(self.config.get("test_path"))

            self.train_df=yelp_adaptive(self.train_df)
            self.test_df=yelp_adaptive(self.test_df)


            
        # prompt 长度控制相关
        self.max_prompt_length = int(config.get("max_prompt_length", 1024))
        self.truncation = config.get("truncation", "error")  # 'left' / 'right' / 'middle' / 'error'

        # 其它选项
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed", None)

        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.tokenizer = tokenizer
        self.config = config
        self._build_samples()


    def _build_samples(self):
        if self.is_train:
            df = self.train_df
        else:
            df = self.test_df

        # df=df.iloc[:1000]  # for debug
        processed = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="[YelpDuetDataset] Building samples"):
    
            uid, iid = row['user_id'], row['item_id']
            user_title = row.get('reviewerName', f"User_{uid}")
            item_title = row.get('title', f"Item_{iid}")
            gt_rating = float(row['ratings'])
            current_time = row['unixReviewTime']  # 关键：当前样本的时间戳，用于过滤历史
            
            # 1. 严格过滤用户历史（仅保留当前时间之前的记录）

            user_history = df[
                (df['user_id'] == uid) & 
                (df['unixReviewTime'] < current_time)  # 仅历史数据
            ]
            if self.is_train is False:
                train_df=self.train_df
                user_history = pd.concat(
                [
                    user_history,
                    train_df[(train_df['user_id'] == uid)& (train_df['unixReviewTime'] < current_time)]
                    
                ] )
            user_history = user_history.sort_values('unixReviewTime').tail(10)  # 最近10条
            
            # 2. 严格过滤物品历史（仅保留当前时间之前的记录）
            item_history = df[
                (df['item_id'] == iid) & 
                (df['unixReviewTime'] < current_time)  # 仅历史数据
            ]
            if self.is_train is False:
                train_df=self.train_df
                item_history = pd.concat(
                [
                    item_history,
                    train_df[(train_df['item_id'] == iid)& (train_df['unixReviewTime'] < current_time)]
                    
                ] )
            
            item_history = item_history.sort_values('unixReviewTime').tail(10)  # 最近10条

            if self.is_train:
                if len(user_history) < 1:
                    continue  # 训练时严格要求用户有历史数据，但物品的title可以提供信息
            
            # 3. 处理历史数据缺失的情况
            if user_history.empty:
                user_avg_rating = "N/A (no historical data)"
                user_history_text = "[No historical interactions available for this user]"
            else:
                user_avg_rating = f"{user_history['ratings'].mean():.1f}"
                user_history_text = "\n".join([
                    f"[History {i+1}] Item: {r['title']}, Rating: {r['ratings']:.1f}\nReview: {r.get('reviews', '').strip()[:200]}"
                    for i, (_, r) in enumerate(user_history.iterrows())
                ])
            
            if item_history.empty:
                item_avg_rating = "N/A (no historical data)"
                item_history_text = "[No historical reviews available for this item]"
            else:
                item_avg_rating = f"{item_history['ratings'].mean():.1f}"
                item_history_text = "\n".join([
                    f"[Review {i+1}] User: {r.get('reviewerName', 'Anonymous')}, Rating: {r['ratings']:.1f}\nReview: {r.get('reviews', '').strip()[:200]}"
                    for i, (_, r) in enumerate(item_history.iterrows())
                ])
            
            profile_prompt = DUET_PROFILE_GENERATOR_PROMPT.format(
                user_title=user_title,
                item_title=item_title,
                user_history_text=user_history_text,
                item_history_text=item_history_text,
                user_avg_rating=user_avg_rating,
                item_avg_rating=item_avg_rating
            )
            
            processed.append({
                "messages": [
                    {"role": "system", "content": DUET_PROFILE_SYSTEM_PROMPT},
                    {"role": "user", "content": profile_prompt}
                ],
                "gt_rating": gt_rating,
                "user_title": user_title,
                "item_title": item_title,
                "user_id": uid,
                "item_id": iid,
                "user_avg_rating": user_avg_rating,
                "item_avg_rating": item_avg_rating
            })
        self.samples = processed
        print(f"[DuetDataset] Total samples built: {len(self.samples)}")    


    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.
        
        :param idx: Index of the sample to retrieve.
        :return: The sample's input features for the model.
        """
        row_dict: dict = {}
        sample = self.samples[idx]
        messages = sample["messages"]
        model_inputs = {}
        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        try:
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
        except:
            print("Error in postprocess_data for sample index:", idx, "with  length:", len(input_ids[0]))
            print("===============messages=================")
            print(messages)

            


        # 3) position_ids
        position_ids = compute_position_id_with_mask(attention_mask)
        # 核心输入张量
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        # 4) raw_prompt_ids（不用于输入，只是对齐 RLHFDataset 的接口）
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )


        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings


        # 辅助信息（方便 reward function 用）
        # row_dict["user_id"] = sample["user_id"]
        # row_dict["sess_idx"] = sample["sess_idx"]
        # row_dict["base_profile"] = sample["base_profile"]
        # row_dict["next_session"] = sample["next_session"]

        row_dict["data_source"] ="DUET"
        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()

        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["extra_info"]["messages"] = messages
        # tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        # interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        # need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        # if need_tools_kwargs and not tools_kwargs:
        #     logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        # row_dict["tools_kwargs"] = tools_kwargs
        # row_dict["interaction_kwargs"] = interaction_kwargs
        row_dict["reward_model"] = {}
        row_dict["reward_model"]["ground_truth"] = {
            "user_title": sample["user_title"],
            "item_title": sample["item_title"],
            "user_avg_rating": sample["user_avg_rating"],
            "item_avg_rating": sample["item_avg_rating"],
            "gt_rating": sample["gt_rating"]
        }
        return row_dict
    


        

    def __len__(self):
        return len(self.samples)


class MovieProfileDataset(Dataset):
    """
    Dataset for GRPO-style training of the movie memory controller in Verl.

    和 RLHFDataset 接口风格对齐：
    - __init__(data_files, tokenizer, config, processor=None, max_samples=-1)
      其中 data_files / processor 在这个任务里其实用不上，但保留形参方便 Trainer 复用。
    - 内部从 user_logs + user_memories 的 JSON 构造样本，不依赖 parquet。

    每个样本对应一个:
        (uid, sess_idx, base_profile, session_now, next_session)

    __getitem__ 返回:
        - input_ids, attention_mask, position_ids
        - raw_prompt_ids
        - 元信息: uid, sess_idx, base_profile, next_session, index
    """

    def __init__(
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[object] = None,
        max_samples: int = -1,
        is_train=True
    ):
        # data_files/processor 在本任务中不用，但保留参数以兼容 Verl 的调用方式
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]

        self.data_files = data_files
        self.tokenizer = tokenizer
        self.processor = None  # 电影日志是纯文本，这里不支持多模态
        self.max_samples = max_samples
        self.config = config

        # 路径相关配置
        if is_train:
            self.user_log_path = config.get("train_user_log_path", None)
        else:
            self.user_log_path = config.get("val_user_log_path", None)

        print(f"==========================is_train = {is_train}==========================")
        print(f"load data from {self.user_log_path}")

        # self.user_memory_path = os.path.expanduser(config.get("user_memory_path"))

        # 采样 / session 切分相关
        self.user_num = config.get("user_num", None)
        self.x_hundred_as_short_term_memory = float(config.get("x_hundred_as_short_term_memory", 1.0))
        self.session_length = int(config.get("session_length", 8))

        # prompt 长度控制相关
        self.max_prompt_length = int(config.get("max_prompt_length", 1024))
        self.truncation = config.get("truncation", "error")  # 'left' / 'right' / 'middle' / 'error'

        # 其它选项
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed", None)

        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        # 预处理：从 JSON 构造样本
        self._build_samples()


    # --------- 从 JSON 构造样本表 --------- #

    def _build_samples(self):
        with open(self.user_log_path, "r", encoding="utf-8") as f:
            user_logs = json.load(f)

        samples = []

        for uid, logs in user_logs.items():
            uid_str = str(uid)

            # -------- 三段切分 --------
            # 70， 2145超过
            profile_logs = logs[-41:-11]
            reward_logs = logs[-11:-1]
            target_log = logs[-1:]

            # -------- 渲染 profile history --------
            logs=[]
            for item in profile_logs:
                tags = item.get("tags",[]) 
                tags_part = f", tags={','.join(tags)}" if tags!=[] else "" 
                log=MOVIE_SESSION_ITEM_TEMPLATE.format( 
                    title=item["title"], 
                    genres=item["genres"],
                    rating=item["rating"], 
                    tags_part=tags_part 
                ) 
                logs.append(log)

            user_prompt = MOVIE_PROFILE_GENERATION_USER_PROMPT.format(
                user_history="\n".join(logs)
            )

            messages = [
                {"role": "system", "content": MOVIE_PROFILE_GENERATION_SYSTEM_PROMPT},   
                {"role": "user", "content": user_prompt},
            ]

            samples.append(
                {
                    "user_id": uid_str,
                    "messages": messages,
                    "session_now": reward_logs,
                    "next_session": target_log,
                }
            )

        self.samples = samples
        print(f"[MovieDataset] Total samples built: {len(self.samples)}")

    # --------- Dataset 接口 --------- #

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        row_dict: dict = {}
        sample = self.samples[idx]
        messages = sample["messages"]

        # print("===============prompt_text=================")
        # print(type(prompt_text))
        # print(prompt_text)

        # 1) 先用 tokenizer 得到 input_ids / attention_mask（不加特殊符号，因为 chat_template 通常已含）

        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        model_inputs = self.tokenizer(
            raw_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        # 2) 用 Verl 的 postprocess 做 left_pad + 截断
        try:
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
        except:
            print(f"===============超长的 user是：{sample['user_id']}===============")
            

        # 3) position_ids
        position_ids = compute_position_id_with_mask(attention_mask)
        # 核心输入张量
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        # 4) raw_prompt_ids（不用于输入，只是对齐 RLHFDataset 的接口）
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )


        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings


        # 辅助信息（方便 reward function 用）
        # row_dict["user_id"] = sample["user_id"]
        # row_dict["sess_idx"] = sample["sess_idx"]
        # row_dict["base_profile"] = sample["base_profile"]
        # row_dict["next_session"] = sample["next_session"]

        row_dict["data_source"] =" MovieMemoryProfile"
        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        row_dict["extra_info"]["user_id"]= sample["user_id"]
        index = row_dict.get("extra_info", {}).get("index", 0)
        # tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        # interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        # need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        # if need_tools_kwargs and not tools_kwargs:
        #     logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        # row_dict["tools_kwargs"] = tools_kwargs
        # row_dict["interaction_kwargs"] = interaction_kwargs
        row_dict["reward_model"] = {}
        row_dict["reward_model"]["ground_truth"] = {
            "session_now": sample["session_now"],
            "next_session": sample["next_session"],
        }
        return row_dict
    

    # 为了和 RLHFDataset 的 checkpoint 行为兼容，简单实现 __getstate__
    def __getstate__(self):
        return self.__dict__.copy()

    # 如果以后你想支持从 ckpt 恢复，可以仿照 RLHFDataset.resume_dataset_state 重新构建 samples
    def resume_dataset_state(self):
        print("[MovieDataset] resume_dataset_state() called, "
              "for now just rebuild samples from JSON.")
        self._build_samples()

class MovieMemoryControllerDataset(Dataset):
    """
    Dataset for GRPO-style training of the movie memory controller in Verl.

    和 RLHFDataset 接口风格对齐：
    - __init__(data_files, tokenizer, config, processor=None, max_samples=-1)
      其中 data_files / processor 在这个任务里其实用不上，但保留形参方便 Trainer 复用。
    - 内部从 user_logs + user_memories 的 JSON 构造样本，不依赖 parquet。

    每个样本对应一个:
        (uid, sess_idx, base_profile, session_now, next_session)

    __getitem__ 返回:
        - input_ids, attention_mask, position_ids
        - raw_prompt_ids
        - 元信息: uid, sess_idx, base_profile, next_session, index
    """

    def __init__(
        self,
        data_files: str | list[str],  # 为了兼容 RLHFDataset 的签名，这里不实际使用
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[object] = None,
        max_samples: int = -1,
    ):
        # data_files/processor 在本任务中不用，但保留参数以兼容 Verl 的调用方式
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]

        self.data_files = data_files
        self.tokenizer = tokenizer
        self.processor = None  # 电影日志是纯文本，这里不支持多模态
        self.max_samples = max_samples
        self.config = config

        # 路径相关配置
        self.user_log_path = os.path.expanduser(config.get("user_log_path"))
        self.user_memory_path = os.path.expanduser(config.get("user_memory_path"))

        # 采样 / session 切分相关
        self.user_num = config.get("user_num", None)
        self.x_hundred_as_short_term_memory = float(config.get("x_hundred_as_short_term_memory", 1.0))
        self.session_length = int(config.get("session_length", 8))

        # prompt 长度控制相关
        self.max_prompt_length = int(config.get("max_prompt_length", 1024))
        self.truncation = config.get("truncation", "error")  # 'left' / 'right' / 'middle' / 'error'

        # 其它选项
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed", None)

        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        # 预处理：从 JSON 构造样本
        self._build_samples()

    # --------- 辅助：构造 session 文本 & 控制器 prompt --------- #

    @staticmethod
    def _render_session_text(session):
        """把一个 session 的多条电影日志按模板转成文本。"""
        return "\n".join(
            [
                MOVIE_SESSION_ITEM_TEMPLATE.format(
                    title=item["title"],
                    genres=item["genres"],
                    rating=item["rating"],
                    tags=", ".join(item["tags"]),
                )
                for item in session
            ]
        )

    def _build_controller_prompt(self,profile: str, session):
        """构造 Memory Controller 的 user prompt 文本。"""
        session_text = self._render_session_text(session)
        prompt = MOVIE_MEMORY_CONTROLLER_USER_PROMPT.format(
            base_profile=profile,
            session_text=session_text,
        )
        return prompt

    # --------- 从 JSON 构造样本表 --------- #

    def _build_samples(self):
        with open(self.user_log_path, "r", encoding="utf-8") as f:
            user_logs = json.load(f)

        with open(self.user_memory_path, "r", encoding="utf-8") as f:
            user_memories = json.load(f)

        print(
            f"[MovieDataset] Loaded {len(user_logs)} user logs from {self.user_log_path} "
            f"and {len(user_memories)} user memories from {self.user_memory_path}."
        )

        samples = []


        # 遍历用户
        for uid, logs in user_logs.items():


            uid_str = str(uid)
            if uid_str not in user_memories:
                print(f"[MovieDataset Warning] uid {uid_str} not in user_memories, skip.")
                continue

            merged_summary = user_memories[uid_str]["merged_summary"]
            print(f"================== User {uid_str} merged summary ==================")
            print(merged_summary)

            # 取后 x*100 条日志作为短期部分
            last = logs[-int(self.x_hundred_as_short_term_memory * 100) -1:]

            last_l = len(last)
            print(f"[MovieDataset] User {uid_str} last logs count: {last_l}")

            # 按 session_length 切 session
            sessions = []
            for i in range(0, last_l, self.session_length):
                sessions.append(last[i : i + self.session_length])

            print(f"[MovieDataset] User {uid_str} has {len(sessions)} sessions.")

            # 每个 (session_now, next_session) 做一个训练样本
            for sess_idx in range(len(sessions) - 1):
                session_now = sessions[sess_idx]
                next_session = sessions[sess_idx + 1]

                system_prompt=MOVIE_MEMORY_CONTROLLER_SYSTEM_PROMPT
                user_prompt = self._build_controller_prompt(
                    profile=merged_summary,
                    session=session_now,
                )
                messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                # raw_prompt = self.tokenizer.apply_chat_template(
                #     messages,
                #     tokenize=False,
                #     add_generation_prompt=True,
                #     enable_thinking=False,
                # )
                # print("===============prompt_text=================")
                # print(raw_prompt)

                samples.append(
                    {
                        "user_id": uid_str,
                        # "sess_idx": sess_idx,
                        "messages": messages,
                        # "base_profile": merged_summary,
                        # "session_now": session_now,
                        "next_session": next_session,
                        # extra_info 用于 index 等扩展信息（和 RLHFDataset 对齐）
                    }
                )

        # 如果需要 max_samples 和 shuffle，在这里下手
        total = len(samples)
        if self.max_samples > 0 and self.max_samples < total:
            indices = np.arange(total)
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                rng.shuffle(indices)
            indices = indices[: self.max_samples]
            samples = [samples[i] for i in indices]
            print(f"[MovieDataset] selected {self.max_samples} samples out of {total}")

        self.samples = samples
        print(f"[MovieDataset] Total samples built: {len(self.samples)}")

    # --------- Dataset 接口 --------- #

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        row_dict: dict = {}
        sample = self.samples[idx]
        messages = sample["messages"]


        # 1) 先用 tokenizer 得到 input_ids / attention_mask（不加特殊符号，因为 chat_template 通常已含）
        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        print("===============raw_prompt=================")
        print(raw_prompt)
        model_inputs = self.tokenizer(
            raw_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        # 2) 用 Verl 的 postprocess 做 left_pad + 截断
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # 3) position_ids
        position_ids = compute_position_id_with_mask(attention_mask)
        # 核心输入张量
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        # 4) raw_prompt_ids（不用于输入，只是对齐 RLHFDataset 的接口）
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )


        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings


        # 辅助信息（方便 reward function 用）
        # row_dict["user_id"] = sample["user_id"]
        # row_dict["sess_idx"] = sample["sess_idx"]
        # row_dict["base_profile"] = sample["base_profile"]
        # row_dict["next_session"] = sample["next_session"]

        row_dict["data_source"] ="MovieMemory"
        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        row_dict["extra_info"]["user_id"]= sample["user_id"]
        index = row_dict.get("extra_info", {}).get("index", 0)
        # tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        # interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        # need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        # if need_tools_kwargs and not tools_kwargs:
        #     logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        # row_dict["tools_kwargs"] = tools_kwargs
        # row_dict["interaction_kwargs"] = interaction_kwargs
        row_dict["reward_model"]={}
        row_dict["reward_model"]["ground_truth"]={}
        row_dict["reward_model"]["ground_truth"]["next_session"] = sample["next_session"]
        return row_dict
    

    # 为了和 RLHFDataset 的 checkpoint 行为兼容，简单实现 __getstate__
    def __getstate__(self):
        return self.__dict__.copy()

    # 如果以后你想支持从 ckpt 恢复，可以仿照 RLHFDataset.resume_dataset_state 重新构建 samples
    def resume_dataset_state(self):
        print("[MovieDataset] resume_dataset_state() called, "
              "for now just rebuild samples from JSON.")
        self._build_samples()