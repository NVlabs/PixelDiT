# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

# Modified from https://github.com/NVlabs/Sana

import getpass
import hashlib
import io
import json
import os
import os.path as osp
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image
from termcolor import colored
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

from diffusion.data.builder import DATASETS
from diffusion.data.datasets.utils import *
from diffusion.data.wids import ShardListDataset, ShardListDatasetMulti, lru_json_load
from diffusion.utils.logger import get_root_logger


# --------------------------------------------------------------------------
# Base pixel-space datasets
# --------------------------------------------------------------------------
@DATASETS.register_module()
class PixDiTImageDataset(Dataset):
    """Simple image + caption dataset (pixel space)."""

    def __init__(
        self,
        data_dir="",
        transform=None,
        resolution=256,
        load_vae_feat=False,
        load_text_feat=False,
        max_length=300,
        config=None,
        caption_proportion=None,
        external_caption_suffixes=None,
        external_clipscore_suffixes=None,
        clip_thr=0.0,
        clip_thr_temperature=1.0,
        img_extension=".png",
        **kwargs,
    ):
        if external_caption_suffixes is None:
            external_caption_suffixes = []
        if external_clipscore_suffixes is None:
            external_clipscore_suffixes = []

        self.logger = (
            get_root_logger() if config is None else get_root_logger(osp.join(config.work_dir, "train_log.log"))
        )
        self.transform = transform if not load_vae_feat else None
        self.load_vae_feat = load_vae_feat
        self.load_text_feat = load_text_feat
        self.resolution = resolution
        self.max_length = max_length
        self.caption_proportion = caption_proportion if caption_proportion is not None else {"prompt": 1.0}
        self.external_caption_suffixes = external_caption_suffixes
        self.external_clipscore_suffixes = external_clipscore_suffixes
        self.clip_thr = clip_thr
        self.clip_thr_temperature = clip_thr_temperature
        self.default_prompt = "prompt"
        self.img_extension = img_extension

        self.data_dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        self.dataset = []
        for data_dir in self.data_dirs:
            meta_data = json.load(open(osp.join(data_dir, "meta_data.json")))
            self.dataset.extend([osp.join(data_dir, i) for i in meta_data["img_names"]])

        self.dataset = self.dataset * 2000
        self.logger.info(colored("Dataset is repeat 2000 times for toy dataset", "red", attrs=["bold"]))
        self.ori_imgs_nums = len(self)
        self.logger.info(f"Dataset samples: {len(self.dataset)}")

        self.logger.info(f"Loading external caption json from: original_filename{external_caption_suffixes}.json")
        self.logger.info(f"Loading external clipscore json from: original_filename{external_clipscore_suffixes}.json")
        self.logger.info(f"external caption clipscore threshold: {clip_thr}, temperature: {clip_thr_temperature}")
        self.logger.info(f"Text max token length: {self.max_length}")

    def weighted_sample_clipscore(self, data, info):
        labels = []
        weights = []
        fallback_label = None
        max_clip_score = float("-inf")

        for suffix in self.external_clipscore_suffixes:
            clipscore_json_path = f"{data}{suffix}.json"

            if os.path.exists(clipscore_json_path):
                try:
                    clipscore_json = lru_json_load(clipscore_json_path)
                except Exception:
                    clipscore_json = {}
                if self.key in clipscore_json:
                    clip_scores = clipscore_json[self.key]

                    for caption_type, clip_score in clip_scores.items():
                        clip_score = float(clip_score)
                        if caption_type in info:
                            if clip_score >= self.clip_thr:
                                labels.append(caption_type)
                                weights.append(clip_score)

                            if clip_score > max_clip_score:
                                max_clip_score = clip_score
                                fallback_label = caption_type

        if not labels and fallback_label:
            return fallback_label, max_clip_score

        if not labels:
            return self.default_prompt, 0.0

        adjusted_weights = np.array(weights) ** (1.0 / max(self.clip_thr_temperature, 0.01))
        normalized_weights = adjusted_weights / np.sum(adjusted_weights)
        sampled_label = random.choices(labels, weights=normalized_weights, k=1)[0]
        index = labels.index(sampled_label)
        original_weight = weights[index]

        return sampled_label, original_weight

    def getdata(self, idx):
        data = self.dataset[idx]
        img_extensions = [".jpg", ".png", ".jpeg", ".webp"]
        filename, ext = os.path.splitext(data)
        if ext in img_extensions:
            data = filename
            self.img_extension = ext
        self.key = data.split("/")[-1]
        info = {}
        with open(f"{data}.txt") as f:
            info[self.default_prompt] = f.readlines()[0].strip()

        for suffix in self.external_caption_suffixes:
            caption_json_path = f"{data}{suffix}.json"
            if os.path.exists(caption_json_path):
                try:
                    caption_json = lru_json_load(caption_json_path)
                except Exception:
                    caption_json = {}
                if self.key in caption_json:
                    info.update(caption_json[self.key])

        caption_type, caption_clipscore = self.weighted_sample_clipscore(data, info)
        caption_type = caption_type if caption_type in info else self.default_prompt
        txt_fea = "" if info[caption_type] is None else info[caption_type]

        data_info = {
            "img_hw": torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
            "aspect_ratio": torch.tensor(1.0),
        }

        if self.load_vae_feat:
            raise ValueError("Load VAE is not supported now")
        img_path = f"{data}{self.img_extension}"
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)

        attention_mask = torch.ones(1, 1, self.max_length, dtype=torch.int16)
        if self.load_text_feat:
            npz_path = f"{self.key}.npz"
            txt_info = np.load(npz_path)
            txt_fea = torch.from_numpy(txt_info["caption_feature"])
            if "attention_mask" in txt_info:
                attention_mask = torch.from_numpy(txt_info["attention_mask"])[None]
            if txt_fea.shape[1] != self.max_length:
                txt_fea = torch.cat([txt_fea, txt_fea[:, -1:].repeat(1, self.max_length - txt_fea.shape[1], 1)], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.zeros(1, 1, self.max_length - attention_mask.shape[-1])], dim=-1
                )

        return (
            img,
            txt_fea,
            attention_mask.to(torch.int16),
            data_info,
            idx,
            caption_type,
            "",
            str(caption_clipscore),
        )

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = idx + 1
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.dataset)


@DATASETS.register_module()
class PixDiTWebDataset(Dataset):
    """WebDataset-based pixel-space loader."""

    def __init__(
        self,
        data_dir="",
        meta_path=None,
        cache_dir="~/.cache/pixdit-webds-meta",
        max_shards_to_load=None,
        transform=None,
        resolution=256,
        load_vae_feat=False,
        load_text_feat=False,
        max_length=300,
        config=None,
        caption_proportion=None,
        sort_dataset=False,
        num_replicas=None,
        external_caption_suffixes=None,
        external_clipscore_suffixes=None,
        clip_thr=0.0,
        clip_thr_temperature=1.0,
        **kwargs,
    ):
        if external_caption_suffixes is None:
            external_caption_suffixes = []
        if external_clipscore_suffixes is None:
            external_clipscore_suffixes = []

        self.logger = (
            get_root_logger() if config is None else get_root_logger(osp.join(config.work_dir, "train_log.log"))
        )
        self.transform = transform if not load_vae_feat else None
        self.load_vae_feat = load_vae_feat
        self.load_text_feat = load_text_feat
        self.resolution = resolution
        self.max_length = max_length
        self.caption_proportion = caption_proportion if caption_proportion is not None else {"prompt": 1.0}
        self.external_caption_suffixes = external_caption_suffixes
        self.external_clipscore_suffixes = external_clipscore_suffixes
        self.clip_thr = clip_thr
        self.clip_thr_temperature = clip_thr_temperature
        self.default_prompt = "prompt"

        data_dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        meta_paths = meta_path if isinstance(meta_path, list) else [meta_path] * len(data_dirs)
        self.meta_paths = []
        for data_path, meta_path in zip(data_dirs, meta_paths):
            self.data_path = osp.expanduser(data_path)
            self.meta_path = osp.expanduser(meta_path) if meta_path is not None else None

            _local_meta_path = osp.join(self.data_path, "wids-meta.json")
            if meta_path is None and osp.exists(_local_meta_path):
                self.logger.info(f"loading from {_local_meta_path}")
                self.meta_path = meta_path = _local_meta_path

            if meta_path is None:
                self.meta_path = osp.join(
                    osp.expanduser(cache_dir),
                    self.data_path.replace("/", "--") + f".max_shards:{max_shards_to_load}" + ".wdsmeta.json",
                )

            assert osp.exists(self.meta_path), f"meta path not found in [{self.meta_path}] or [{_local_meta_path}]"
            self.logger.info(f"Loading meta information {self.meta_path}")
            self.meta_paths.append(self.meta_path)

        self._initialize_dataset(num_replicas, sort_dataset)

        self.logger.info(f"Loading external caption json from: original_filename{external_caption_suffixes}.json")
        self.logger.info(f"Loading external clipscore json from: original_filename{external_clipscore_suffixes}.json")
        self.logger.info(f"external caption clipscore threshold: {clip_thr}, temperature: {clip_thr_temperature}")
        self.logger.info(f"Text max token length: {self.max_length}")
        self.logger.warning(f"Sort the dataset: {sort_dataset}")

    def _initialize_dataset(self, num_replicas, sort_dataset):
        uuid = hashlib.sha256(self.meta_path.encode()).hexdigest()[:8]
        if len(self.meta_paths) > 0:
            self.dataset = ShardListDatasetMulti(
                self.meta_paths,
                cache_dir=osp.expanduser(f"~/.cache/_wids_cache/{getpass.getuser()}-{uuid}"),
                sort_data_inseq=sort_dataset,
                num_replicas=num_replicas or dist.get_world_size(),
            )
        else:
            self.dataset = ShardListDataset(
                self.meta_path, cache_dir=osp.expanduser(f"~/.cache/_wids_cache/{getpass.getuser()}-{uuid}")
            )
        self.ori_imgs_nums = len(self)
        self.logger.info(f"{self.dataset.data_info}")

    def weighted_sample_clipscore(self, data, info):
        labels = []
        weights = []
        fallback_label = None
        max_clip_score = float("-inf")

        for suffix in self.external_clipscore_suffixes:
            clipscore_json_path = data["__shard__"].replace(".tar", f"{suffix}.json")

            if os.path.exists(clipscore_json_path):
                try:
                    clipscore_json = lru_json_load(clipscore_json_path)
                except Exception:
                    clipscore_json = {}
                if self.key in clipscore_json:
                    clip_scores = clipscore_json[self.key]

                    for caption_type, clip_score in clip_scores.items():
                        clip_score = float(clip_score)
                        if caption_type in info:
                            if clip_score >= self.clip_thr:
                                labels.append(caption_type)
                                weights.append(clip_score)

                            if clip_score > max_clip_score:
                                max_clip_score = clip_score
                                fallback_label = caption_type

        if not labels and fallback_label:
            return fallback_label, max_clip_score

        if not labels:
            return self.default_prompt, 0.0

        adjusted_weights = np.array(weights) ** (1.0 / max(self.clip_thr_temperature, 0.01))
        normalized_weights = adjusted_weights / np.sum(adjusted_weights)
        sampled_label = random.choices(labels, weights=normalized_weights, k=1)[0]
        index = labels.index(sampled_label)
        original_weight = weights[index]

        return sampled_label, original_weight

    def getdata(self, idx):
        data = self.dataset[idx]
        info = data[".json"]
        self.key = data["__key__"]
        dataindex_info = {
            "index": data["__index__"],
            "shard": "/".join(data["__shard__"].rsplit("/", 2)[-2:]),
            "shardindex": data["__shardindex__"],
        }

        for suffix in self.external_caption_suffixes:
            caption_json_path = data["__shard__"].replace(".tar", f"{suffix}.json")
            if os.path.exists(caption_json_path):
                try:
                    caption_json = lru_json_load(caption_json_path)
                except Exception:
                    caption_json = {}
                if self.key in caption_json:
                    info.update(caption_json[self.key])

        caption_type, caption_clipscore = self.weighted_sample_clipscore(data, info)
        caption_type = caption_type if caption_type in info else self.default_prompt
        txt_fea = "" if info[caption_type] is None else info[caption_type]

        data_info = {
            "img_hw": torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
            "aspect_ratio": torch.tensor(1.0),
        }

        if self.load_vae_feat:
            img = data[".npy"]
        else:
            img = data[".png"] if ".png" in data else data[".jpg"]
        if self.transform:
            img = self.transform(img)

        attention_mask = torch.ones(1, 1, self.max_length, dtype=torch.int16)
        if self.load_text_feat:
            npz_path = f"{self.key}.npz"
            txt_info = np.load(npz_path)
            txt_fea = torch.from_numpy(txt_info["caption_feature"])
            if "attention_mask" in txt_info:
                attention_mask = torch.from_numpy(txt_info["attention_mask"])[None]
            if txt_fea.shape[1] != self.max_length:
                txt_fea = torch.cat([txt_fea, txt_fea[:, -1:].repeat(1, self.max_length - txt_fea.shape[1], 1)], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.zeros(1, 1, self.max_length - attention_mask.shape[-1])], dim=-1
                )

        return (
            img,
            txt_fea,
            attention_mask.to(torch.int16),
            data_info,
            idx,
            caption_type,
            dataindex_info,
            str(caption_clipscore),
        )

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = idx + 1
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.dataset)

    def get_data_info(self, idx):
        try:
            data = self.dataset[idx]
            info = data[".json"]
            key = data["__key__"]
            version = info.get("version", "others")
            return {"height": info["height"], "width": info["width"], "version": version, "key": key}
        except Exception as e:
            print(f"Error details: {str(e)}")
            return None


@DATASETS.register_module()
class PixDiTWebDatasetMS(PixDiTWebDataset):
    """Multi-scale WebDataset with aspect-ratio bucketing."""

    def __init__(
        self,
        data_dir="",
        meta_path=None,
        cache_dir="~/.cache/pixdit-webds-meta",
        max_shards_to_load=None,
        transform=None,
        resolution=256,
        sample_subset=None,
        load_vae_feat=False,
        load_text_feat=False,
        input_size=32,
        patch_size=2,
        max_length=300,
        config=None,
        caption_proportion=None,
        sort_dataset=False,
        num_replicas=None,
        external_caption_suffixes=None,
        external_clipscore_suffixes=None,
        clip_thr=0.0,
        clip_thr_temperature=1.0,
        vae_downsample_rate=32,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            meta_path=meta_path,
            cache_dir=cache_dir,
            max_shards_to_load=max_shards_to_load,
            transform=transform,
            resolution=resolution,
            sample_subset=sample_subset,
            load_vae_feat=load_vae_feat,
            load_text_feat=load_text_feat,
            input_size=input_size,
            patch_size=patch_size,
            max_length=max_length,
            config=config,
            caption_proportion=caption_proportion,
            sort_dataset=sort_dataset,
            num_replicas=num_replicas,
            external_caption_suffixes=external_caption_suffixes,
            external_clipscore_suffixes=external_clipscore_suffixes,
            clip_thr=clip_thr,
            clip_thr_temperature=clip_thr_temperature,
            vae_downsample_rate=32,
            **kwargs,
        )
        aspect_ratio_type = kwargs.pop("aspect_ratio_type")
        self.base_size = int(aspect_ratio_type.split("_")[-1])
        self.aspect_ratio = eval(aspect_ratio_type)
        self.ratio_index = {}
        self.ratio_nums = {}
        self.interpolate_model = (
            InterpolationMode.BICUBIC
            if aspect_ratio_type not in ["ASPECT_RATIO_2048", "ASPECT_RATIO_2880"]
            else InterpolationMode.LANCZOS
        )

        for k, v in self.aspect_ratio.items():
            self.ratio_index[float(k)] = []
            self.ratio_nums[float(k)] = 0

        self.vae_downsample_rate = vae_downsample_rate

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = random.choice(self.ratio_index[self.closest_ratio])
        raise RuntimeError("Too many bad data.")

    def getdata(self, idx):
        data = self.dataset[idx]
        info = data[".json"]
        self.key = data["__key__"]
        dataindex_info = {
            "index": data["__index__"],
            "shard": "/".join(data["__shard__"].rsplit("/", 2)[-2:]),
            "shardindex": data["__shardindex__"],
        }

        for suffix in self.external_caption_suffixes:
            caption_json_path = data["__shard__"].replace(".tar", f"{suffix}.json")
            if os.path.exists(caption_json_path):
                try:
                    caption_json = lru_json_load(caption_json_path)
                except Exception:
                    caption_json = {}
                if self.key in caption_json:
                    info.update(caption_json[self.key])

        data_info = {}
        ori_h, ori_w = info["height"], info["width"]

        closest_size, closest_ratio = min(
            ((v, float(k)) for k, v in self.aspect_ratio.items()),
            key=lambda kv: abs(float(kv[1]) - ori_h / ori_w),
        )
        closest_size = list(map(lambda x: int(x), closest_size))
        self.closest_ratio = closest_ratio

        data_info["img_hw"] = torch.tensor([ori_h, ori_w], dtype=torch.float32)
        data_info["aspect_ratio"] = closest_ratio

        caption_type, caption_clipscore = self.weighted_sample_clipscore(data, info)
        caption_type = caption_type if caption_type in info else self.default_prompt
        txt_fea = "" if info[caption_type] is None else info[caption_type]

        if self.load_vae_feat:
            img = data[".npy"]
            if len(img.shape) == 4 and img.shape[0] == 1:
                img = img[0]
            h, w = (img.shape[1], img.shape[2])
            assert h == int(closest_size[0] // self.vae_downsample_rate) and w == int(
                closest_size[1] // self.vae_downsample_rate
            ), f"h: {h}, w: {w}, ori_hw: {closest_size}, data_info: {dataindex_info}"
        else:
            img = data[".png"] if ".png" in data else data[".jpg"]
            if closest_size[0] / ori_h > closest_size[1] / ori_w:
                resize_size = closest_size[0], int(ori_w * closest_size[0] / ori_h)
            else:
                resize_size = int(ori_h * closest_size[1] / ori_w), closest_size[1]
            self.transform = T.Compose(
                [
                    T.Lambda(lambda img: img.convert("RGB")),
                    T.Resize(resize_size, interpolation=self.interpolate_model),
                    T.CenterCrop(closest_size),
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5]),
                ]
            )
        if idx not in self.ratio_index[closest_ratio]:
            self.ratio_index[closest_ratio].append(idx)

        if self.transform:
            img = self.transform(img)

        attention_mask = torch.ones(1, 1, self.max_length, dtype=torch.int16)
        if self.load_text_feat:
            npz_path = f"{self.key}.npz"
            txt_info = np.load(npz_path)
            txt_fea = torch.from_numpy(txt_info["caption_feature"])
            if "attention_mask" in txt_info:
                attention_mask = torch.from_numpy(txt_info["attention_mask"])[None]
            if txt_fea.shape[1] != self.max_length:
                txt_fea = torch.cat([txt_fea, txt_fea[:, -1:].repeat(1, self.max_length - txt_fea.shape[1], 1)], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.zeros(1, 1, self.max_length - attention_mask.shape[-1])], dim=-1
                )

        return (
            img,
            txt_fea,
            attention_mask.to(torch.int16),
            data_info,
            idx,
            caption_type,
            dataindex_info,
            str(caption_clipscore),
        )

    def __len__(self):
        return len(self.dataset)


# --------------------------------------------------------------------------
# Pixel-space variants (wrapper over PixDiTWebDataset)
# --------------------------------------------------------------------------
@DATASETS.register_module(name="PixelDataset")
class PixelDataset(PixDiTWebDataset):
    """Pixel-space variant of the PixDiT web dataset (no VAE latents)."""

    def __init__(self, sort_dataset: bool = False, **kwargs):
        if "data_dir" in kwargs:
            data_dir = kwargs["data_dir"]
            if isinstance(data_dir, str) and data_dir.startswith("[") and data_dir.endswith("]"):
                data_dir = data_dir[1:-1]
                kwargs["data_dir"] = [path.strip() for path in data_dir.split(",")]

        kwargs["load_vae_feat"] = False
        super().__init__(sort_dataset=sort_dataset, **kwargs)

        self.pixel_transforms = T.Compose(
            [
                T.Resize(self.resolution, interpolation=InterpolationMode.BILINEAR),
                T.CenterCrop(self.resolution),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def getdata(self, idx):
        parent_data = super().getdata(idx)
        data = self.dataset[idx]
        img_bytes = data[".png"] if ".png" in data else data[".jpg"]

        if isinstance(img_bytes, Image.Image):
            img = img_bytes.convert("RGB")
        else:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        pixel_tensor = self.pixel_transforms(img)
        return (pixel_tensor,) + parent_data[1:]


@DATASETS.register_module(name="PixelDatasetMS")
class PixelDatasetMS(PixDiTWebDataset):
    """
    Multi-scale pixel-space dataset.
    - Pixel-space processing (no VAE)
    - Provides aspect-ratio buckets required by multi-scale batch sampler
    - Resizes and center-crops per-sample to the closest target size bucket
    """

    def __init__(self, sort_dataset: bool = False, aspect_ratio_type: str = "ASPECT_RATIO_512", **kwargs):
        if "data_dir" in kwargs:
            data_dir = kwargs["data_dir"]
            if isinstance(data_dir, str) and data_dir.startswith("[") and data_dir.endswith("]"):
                data_dir = data_dir[1:-1]
                kwargs["data_dir"] = [path.strip() for path in data_dir.split(",")]

        kwargs["load_vae_feat"] = False
        super().__init__(sort_dataset=sort_dataset, **kwargs)

        self.base_size = int(aspect_ratio_type.split("_")[-1]) if "_" in aspect_ratio_type else 512
        self.aspect_ratio = eval(aspect_ratio_type)

        self.ratio_index = {}
        self.ratio_nums = {}
        for k, _ in self.aspect_ratio.items():
            self.ratio_index[float(k)] = []
            self.ratio_nums[float(k)] = 0

        self.interpolate_model = (
            InterpolationMode.BICUBIC
            if aspect_ratio_type not in ["ASPECT_RATIO_2048", "ASPECT_RATIO_2880"]
            else InterpolationMode.LANCZOS
        )

        self.closest_ratio = 1.0

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                return self.getdata(idx)
            except Exception:
                bucket = self.ratio_index.get(self.closest_ratio, [])
                if bucket:
                    import random as _random

                    idx = _random.choice(bucket)
                else:
                    idx = idx + 1
        raise RuntimeError("Too many bad data.")

    def getdata(self, idx):
        data = self.dataset[idx]
        info = data[".json"]
        self.key = data["__key__"]
        dataindex_info = {
            "index": data["__index__"],
            "shard": "/".join(data["__shard__"].rsplit("/", 2)[-2:]),
            "shardindex": data["__shardindex__"],
        }

        for suffix in self.external_caption_suffixes:
            caption_json_path = data["__shard__"].replace(".tar", f"{suffix}.json")
            if os.path.exists(caption_json_path):
                try:
                    caption_json = lru_json_load(caption_json_path)
                except Exception:
                    caption_json = {}
                if self.key in caption_json:
                    info.update(caption_json[self.key])

        ori_h, ori_w = info["height"], info["width"]
        closest_size, closest_ratio = min(
            ((v, float(k)) for k, v in self.aspect_ratio.items()),
            key=lambda kv: abs(float(kv[1]) - ori_h / ori_w),
        )
        closest_size = [int(closest_size[0]), int(closest_size[1])]
        self.closest_ratio = closest_ratio

        data_info = {
            "img_hw": torch.tensor([ori_h, ori_w], dtype=torch.float32),
            "aspect_ratio": closest_ratio,
        }

        caption_type, caption_clipscore = self.weighted_sample_clipscore(data, info)
        caption_type = caption_type if caption_type in info else self.default_prompt
        txt_fea = "" if info[caption_type] is None else info[caption_type]

        img_bytes = data[".png"] if ".png" in data else data[".jpg"]
        if isinstance(img_bytes, Image.Image):
            img = img_bytes.convert("RGB")
        else:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if ori_h == closest_size[0] and ori_w == closest_size[1]:
            transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            if closest_size[0] / ori_h > closest_size[1] / ori_w:
                resize_size = (closest_size[0], int(ori_w * closest_size[0] / ori_h))
            else:
                resize_size = (int(ori_h * closest_size[1] / ori_w), closest_size[1])
            transform = T.Compose(
                [
                    T.Resize(resize_size, interpolation=self.interpolate_model),
                    T.CenterCrop((closest_size[0], closest_size[1])),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        img = transform(img)

        if idx not in self.ratio_index[closest_ratio]:
            self.ratio_index[closest_ratio].append(idx)

        attention_mask = torch.ones(1, 1, self.max_length, dtype=torch.int16)

        if self.load_text_feat:
            npz_path = f"{self.key}.npz"
            try:
                import numpy as _np

                txt_info = _np.load(npz_path)
                txt_fea = torch.from_numpy(txt_info["caption_feature"])
                if "attention_mask" in txt_info:
                    attention_mask = torch.from_numpy(txt_info["attention_mask"])[None]
                if txt_fea.shape[1] != self.max_length:
                    pad = txt_fea[:, -1:].repeat(1, self.max_length - txt_fea.shape[1], 1)
                    txt_fea = torch.cat([txt_fea, pad], dim=1)
                    attention_mask = torch.cat(
                        [attention_mask, torch.zeros(1, 1, self.max_length - attention_mask.shape[-1])], dim=-1
                    )
            except Exception:
                pass

        return (
            img,
            txt_fea,
            attention_mask.to(torch.int16),
            data_info,
            idx,
            caption_type,
            dataindex_info,
            str(caption_clipscore),
        )



