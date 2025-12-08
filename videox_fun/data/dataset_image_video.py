import csv
import gc
import io
import json
import math
import os
import random
import re
from contextlib import contextmanager
from random import shuffle
from threading import Thread

import albumentations
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader
from einops import rearrange
from func_timeout import FunctionTimedOut, func_timeout
from packaging import version as pver
from PIL import Image
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset

VIDEO_READER_TIMEOUT = 20

def get_random_mask(shape, image_start_only=False):
    f, c, h, w = shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

    if not image_start_only:
        if f != 1:
            mask_index = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.05, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05]) 
        else:
            mask_index = np.random.choice([0, 1], p = [0.2, 0.8])
        if mask_index == 0:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)
            mask[:, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 1:
            mask[:, :, :, :] = 1
        elif mask_index == 2:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:, :, :, :] = 1
        elif mask_index == 3:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:-mask_frame_index, :, :, :] = 1
        elif mask_index == 4:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)

            mask_frame_before = np.random.randint(0, f // 2)
            mask_frame_after = np.random.randint(f // 2, f)
            mask[mask_frame_before:mask_frame_after, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 5:
            mask = torch.randint(0, 2, (f, 1, h, w), dtype=torch.uint8)
        elif mask_index == 6:
            num_frames_to_mask = random.randint(1, max(f // 2, 1))
            frames_to_mask = random.sample(range(f), num_frames_to_mask)

            for i in frames_to_mask:
                block_height = random.randint(1, h // 4)
                block_width = random.randint(1, w // 4)
                top_left_y = random.randint(0, h - block_height)
                top_left_x = random.randint(0, w - block_width)
                mask[i, 0, top_left_y:top_left_y + block_height, top_left_x:top_left_x + block_width] = 1
        elif mask_index == 7:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            a = torch.randint(min(w, h) // 8, min(w, h) // 4, (1,)).item()  # 长半轴
            b = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()  # 短半轴

            for i in range(h):
                for j in range(w):
                    if ((i - center_y) ** 2) / (b ** 2) + ((j - center_x) ** 2) / (a ** 2) < 1:
                        mask[:, :, i, j] = 1
        elif mask_index == 8:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            radius = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()
            for i in range(h):
                for j in range(w):
                    if (i - center_y) ** 2 + (j - center_x) ** 2 < radius ** 2:
                        mask[:, :, i, j] = 1
        elif mask_index == 9:
            for idx in range(f):
                if np.random.rand() > 0.5:
                    mask[idx, :, :, :] = 1
        else:
            raise ValueError(f"The mask_index {mask_index} is not define")
    else:
        if f != 1:
            mask[1:, :, :, :] = 1
        else:
            mask[:, :, :, :] = 1
    return mask

class Camera(object):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

def custom_meshgrid(*args):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def get_relative_pose(cam_params):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses

def ray_condition(K, c2w, H, W, device):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

def process_pose_file(pose_file_path, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu', return_poses=False):
    """Modified from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    with open(pose_file_path, 'r') as f:
        poses = f.readlines()

    poses = [pose.strip().split(' ') for pose in poses[1:]]
    cam_params = [[float(x) for x in pose] for pose in poses]
    if return_poses:
        return cam_params
    else:
        cam_params = [Camera(cam_param) for cam_param in cam_params]

        sample_wh_ratio = width / height
        pose_wh_ratio = original_pose_width / original_pose_height  # Assuming placeholder ratios, change as needed

        if pose_wh_ratio > sample_wh_ratio:
            resized_ori_w = height * pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fx = resized_ori_w * cam_param.fx / width
        else:
            resized_ori_h = width / pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fy = resized_ori_h * cam_param.fy / height

        intrinsic = np.asarray([[cam_param.fx * width,
                                cam_param.fy * height,
                                cam_param.cx * width,
                                cam_param.cy * height]
                                for cam_param in cam_params], dtype=np.float32)

        K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
        c2ws = get_relative_pose(cam_params)  # Assuming this function is defined elsewhere
        c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
        plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
        plucker_embedding = plucker_embedding[None]
        plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
        return plucker_embedding

def process_pose_params(cam_params, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu'):
    """Modified from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    cam_params = [Camera(cam_param) for cam_param in cam_params]

    sample_wh_ratio = width / height
    pose_wh_ratio = original_pose_width / original_pose_height  # Assuming placeholder ratios, change as needed

    if pose_wh_ratio > sample_wh_ratio:
        resized_ori_w = height * pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fx = resized_ori_w * cam_param.fx / width
    else:
        resized_ori_h = width / pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fy = resized_ori_h * cam_param.fy / height

    intrinsic = np.asarray([[cam_param.fx * width,
                            cam_param.fy * height,
                            cam_param.cx * width,
                            cam_param.cy * height]
                            for cam_param in cam_params], dtype=np.float32)

    K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
    c2ws = get_relative_pose(cam_params)  # Assuming this function is defined elsewhere
    c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
    plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
    plucker_embedding = plucker_embedding[None]
    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
    return plucker_embedding

def derive_ground_object_from_instruction(instruction: str) -> str:
    s = (instruction or '').strip()
    if not s:
        return 'the target area'
    s = s.rstrip('.').strip()

    # swap/replace: capture phrase between "replace/swap" and "with/by"
    swap_patterns = [
        r"\breplace\s+(.*?)\s+(?:with|by)\b",
        r"\bswap\s+(.*?)\s+with\b",
    ]
    for pat in swap_patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            phrase = m.group(1).strip(' .,:;')
            if phrase:
                return phrase

    # removal: capture object after remove/delete/erase/eliminate up to a preposition or punctuation
    m = re.search(r"\b(?:remove|delete|erase|eliminate)\s+(.*?)(?:\s+(?:from|in|at|on|over|under|near|by)\b|[.,;]|$)", s, flags=re.IGNORECASE)
    if m:
        phrase = m.group(1).strip(' .,:;')
        if phrase:
            return phrase

    # add/insert: generic target area
    if re.search(r"^\s*(?:add|insert)\b", s, flags=re.IGNORECASE):
        return 'the target area'

    # local style (change/make ...): take the immediate noun after determiner
    m = re.search(r"\b(?:change|make)\s+(?:(the|a|an)\s+)?([A-Za-z][A-Za-z0-9\-]*)", s, flags=re.IGNORECASE)
    if m:
        det = m.group(1) or ''
        noun = m.group(2)
        phrase = (det + ' ' + noun).strip()
        return phrase

    return 'the target area'

class ImageVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket = {'image':[], 'video':[]}

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset.dataset[idx].get('type', 'image')
            self.bucket[content_type].append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket[:]
                del bucket[:]
            elif len(self.bucket['image']) == self.batch_size:
                bucket = self.bucket['image']
                yield bucket[:]
                del bucket[:]

@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()

def get_video_reader_batch(video_reader, batch_index):
    frames = video_reader.get_batch(batch_index).asnumpy()
    return frames

def resize_frame(frame, target_short_side):
    h, w, _ = frame.shape
    if h < w:
        if target_short_side > h:
            return frame
        new_h = target_short_side
        new_w = int(target_short_side * w / h)
    else:
        if target_short_side > w:
            return frame
        new_w = target_short_side
        new_h = int(target_short_side * h / w)
    
    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame

class VideoEditDataset(Dataset):
    def __init__(
        self,
        ann_path, 
        data_root=None,
        video_sample_height: int = None,  # 改为None以支持动态分辨率
        video_sample_width: int = None,   
        video_sample_stride=1, 
        video_sample_n_frames=65,  # 9+8=17 for your case
        source_frames=33,
        edit_frames=32,
        text_drop_ratio=0.1,
        enable_bucket=False,
        enable_inpaint=False,
        instruction_template="A video sequence showing two parts: the first half shows the original scene, and the second half shows the same scene but {edit_instruction}",
    ):
        dataset = json.load(open(ann_path))
        if isinstance(dataset, dict):
            new_dataset = []
            for vid_id, info in dataset.items():
                text_content = info["edit_instruction"]
                new_dataset.append({
                    "original_video": info["original_video"],
                    "edited_video": info["edited_video"],
                    "text": text_content,
                    "type": info.get("type", "video"),
                    # 添加分辨率信息到metadata
                    "resolution": info.get("resolution", None)
                })
            dataset = new_dataset

        self.data_root = data_root
        self.dataset = dataset
        self.length = len(self.dataset)
        
        self.source_frames = source_frames
        self.edit_frames = edit_frames
        self.video_sample_n_frames = video_sample_n_frames
        
        self.instruction_template = instruction_template
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.video_sample_stride = video_sample_stride
        
        # 如果启用bucket，不固定分辨率
        if enable_bucket:
            self.video_sample_height = None
            self.video_sample_width = None
        else:
            self.video_sample_height = video_sample_height
            self.video_sample_width = video_sample_width

    def load_video_pair(self, original_path, edited_path):
        """加载视频对，保持原始分辨率用于bucket training"""
        if self.data_root is not None:
            original_path = os.path.join(self.data_root, original_path)
            edited_path = os.path.join(self.data_root, edited_path)

        with VideoReader_contextmanager(original_path, num_threads=2) as orig_reader, \
             VideoReader_contextmanager(edited_path, num_threads=2) as edit_reader:
            
            # 获取视频信息
            orig_length = len(orig_reader)
            edit_length = len(edit_reader)
            min_length = min(orig_length, edit_length)
            
            # 统一采样策略
            start_idx = 0  # 从头开始
            
            orig_indices = np.linspace(
                start_idx, 
                min(start_idx + (self.source_frames - 1) * self.video_sample_stride, orig_length - 1),
                self.source_frames, 
                dtype=int
            )
            
            edit_indices = np.linspace(
                start_idx,
                min(start_idx + (self.edit_frames - 1) * self.video_sample_stride, edit_length - 1),
                self.edit_frames,
                dtype=int
            )
            
            # 加载帧
            orig_frames = get_video_reader_batch(orig_reader, orig_indices)
            edit_frames = get_video_reader_batch(edit_reader, edit_indices)
            
            # 在拼接前对齐两段视频到相同 HxW（缩放后中心裁剪到 min(H1,H2) x min(W1,W2)）
            def resize_and_center_crop_batch(frames_np, target_h, target_w):
                resized = []
                for i in range(frames_np.shape[0]):
                    frame = frames_np[i]
                    h, w = frame.shape[0], frame.shape[1]
                    scale = max(target_h / h, target_w / w)
                    new_h = int(round(h * scale))
                    new_w = int(round(w * scale))
                    frame_resized = cv2.resize(frame, (new_w, new_h))
                    y0 = max((new_h - target_h) // 2, 0)
                    x0 = max((new_w - target_w) // 2, 0)
                    frame_cropped = frame_resized[y0:y0 + target_h, x0:x0 + target_w]
                    resized.append(frame_cropped)
                return np.stack(resized, axis=0)

            oh, ow = orig_frames.shape[1], orig_frames.shape[2]
            eh, ew = edit_frames.shape[1], edit_frames.shape[2]
            target_h = min(oh, eh)
            target_w = min(ow, ew)
            if (oh != target_h or ow != target_w):
                orig_frames = resize_and_center_crop_batch(orig_frames, target_h, target_w)
            if (eh != target_h or ew != target_w):
                edit_frames = resize_and_center_crop_batch(edit_frames, target_h, target_w)

            # 如果启用bucket，返回numpy数组
            if self.enable_bucket:
                return np.concatenate([orig_frames, edit_frames], axis=0)
            else:
                # 转换为tensor并归一化
                orig_frames = torch.from_numpy(orig_frames).permute(0, 3, 1, 2).contiguous() / 255.
                edit_frames = torch.from_numpy(edit_frames).permute(0, 3, 1, 2).contiguous() / 255.
                return torch.cat([orig_frames, edit_frames], dim=0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        
        while True:
            try:
                # 加载视频对
                pixel_values = self.load_video_pair(
                    data_info['original_video'],
                    data_info['edited_video']
                )
                
                # 准备文本
                text = data_info['text']
                if self.instruction_template and "{edit_instruction}" in self.instruction_template:
                    text = self.instruction_template.format(edit_instruction=text)
                
                if random.random() < self.text_drop_ratio:
                    text = ''
                
                sample = {
                    "pixel_values": pixel_values,
                    "text": text,
                    "data_type": "video",
                    "idx": idx,
                }
                
                # 如果需要inpainting
                if self.enable_inpaint and not self.enable_bucket:
                    # 这里添加inpaint逻辑
                    pass
                
                return sample
                
            except Exception as e:
                try:
                    print(
                        f"Error loading video pair: {e}\n"
                        f"  original={os.path.join(self.data_root, data_info.get('original_video','')) if self.data_root else data_info.get('original_video','')}\n"
                        f"  edited  ={os.path.join(self.data_root, data_info.get('edited_video','')) if self.data_root else data_info.get('edited_video','')}"
                    )
                except Exception:
                    print(f"Error loading video pair: {e}")
                idx = random.randint(0, self.length-1)
    
class VideoEditReasoningDataset(Dataset):
    def __init__(
        self,
        ann_path, 
        data_root=None,
        video_sample_height: int = None,
        video_sample_width: int = None,
        video_sample_stride=1,
        video_sample_n_frames=65,
        source_frames=33,
        reasoning_frames=4,
        edit_frames=32,
        text_drop_ratio=0.1,
        enable_bucket=False,
        enable_inpaint=False,
        instruction_template="A video sequence showing three parts: first the original scene, then grounded {ground_instrction}, and finally the same scene but {edit_instruction}",
    ):
        dataset = json.load(open(ann_path))
        if isinstance(dataset, dict):
            new_dataset = []
            for vid_id, info in dataset.items():
                text_content = info.get("edit_instruction", info.get("text", ""))
                # support both 'grounded_video' and 'ground_video'
                grounded_key = "grounded_video" if "grounded_video" in info else "ground_video"
                new_dataset.append({
                    "original_video": info["original_video"],
                    "grounded_video": info[grounded_key],
                    "edited_video": info["edited_video"],
                    "text": text_content,
                    "edit_instruction": text_content,
                    "type": info.get("type", "video"),
                    "resolution": info.get("resolution", None),
                })
            dataset = new_dataset

        self.data_root = data_root
        self.dataset = dataset
        self.length = len(self.dataset)

        self.source_frames = source_frames
        self.reasoning_frames = reasoning_frames
        self.edit_frames = edit_frames
        self.video_sample_n_frames = video_sample_n_frames

        self.instruction_template = instruction_template
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.video_sample_stride = video_sample_stride

        if enable_bucket:
            self.video_sample_height = None
            self.video_sample_width = None
        else:
            self.video_sample_height = video_sample_height
            self.video_sample_width = video_sample_width

    def load_video_pair(self, original_path, grounded_path, edited_path):
        if self.data_root is not None:
            original_path = os.path.join(self.data_root, original_path)
            grounded_path = os.path.join(self.data_root, grounded_path)
            edited_path = os.path.join(self.data_root, edited_path)

        with VideoReader_contextmanager(original_path, num_threads=2) as orig_reader, \
             VideoReader_contextmanager(grounded_path, num_threads=2) as ground_reader, \
             VideoReader_contextmanager(edited_path, num_threads=2) as edit_reader:

            orig_length = len(orig_reader)
            ground_length = len(ground_reader)
            edit_length = len(edit_reader)

            start_idx = 0

            orig_indices = np.linspace(
                start_idx, 
                min(start_idx + (self.source_frames - 1) * self.video_sample_stride, max(orig_length - 1, 0)),
                self.source_frames, 
                dtype=int
            )

            # reasoning/grounded indices at 8-frame interval (example: 0,7,14,21, ...)
            interval = 8
            ground_indices_full = np.arange(0, max(ground_length, 1), interval, dtype=int)
            if len(ground_indices_full) == 0:
                ground_indices = np.array([0] * self.reasoning_frames, dtype=int)
            else:
                ground_indices = ground_indices_full[: self.reasoning_frames]
                if len(ground_indices) < self.reasoning_frames:
                    pad_value = ground_indices[-1] if len(ground_indices) > 0 else 0
                    ground_indices = np.pad(
                        ground_indices, (0, self.reasoning_frames - len(ground_indices)), constant_values=pad_value
                    )

            edit_indices = np.linspace(
                start_idx,
                min(start_idx + (self.edit_frames - 1) * self.video_sample_stride, max(edit_length - 1, 0)),
                self.edit_frames,
                dtype=int
            )

            orig_frames = get_video_reader_batch(orig_reader, orig_indices)
            ground_frames = get_video_reader_batch(ground_reader, ground_indices)
            edit_frames = get_video_reader_batch(edit_reader, edit_indices)

            def resize_and_center_crop_batch(frames_np, target_h, target_w):
                resized = []
                for i in range(frames_np.shape[0]):
                    frame = frames_np[i]
                    h, w = frame.shape[0], frame.shape[1]
                    scale = max(target_h / h, target_w / w)
                    new_h = int(round(h * scale))
                    new_w = int(round(w * scale))
                    frame_resized = cv2.resize(frame, (new_w, new_h))
                    y0 = max((new_h - target_h) // 2, 0)
                    x0 = max((new_w - target_w) // 2, 0)
                    frame_cropped = frame_resized[y0:y0 + target_h, x0:x0 + target_w]
                    resized.append(frame_cropped)
                return np.stack(resized, axis=0)

            oh, ow = orig_frames.shape[1], orig_frames.shape[2]
            gh, gw = ground_frames.shape[1], ground_frames.shape[2]
            eh, ew = edit_frames.shape[1], edit_frames.shape[2]
            target_h = min(oh, gh, eh)
            target_w = min(ow, gw, ew)
            if (oh != target_h or ow != target_w):
                orig_frames = resize_and_center_crop_batch(orig_frames, target_h, target_w)
            if (gh != target_h or gw != target_w):
                ground_frames = resize_and_center_crop_batch(ground_frames, target_h, target_w)
            if (eh != target_h or ew != target_w):
                edit_frames = resize_and_center_crop_batch(edit_frames, target_h, target_w)

            if self.enable_bucket:
                return np.concatenate([orig_frames, ground_frames, edit_frames], axis=0)
            else:
                orig_frames = torch.from_numpy(orig_frames).permute(0, 3, 1, 2).contiguous() / 255.
                ground_frames = torch.from_numpy(ground_frames).permute(0, 3, 1, 2).contiguous() / 255.
                edit_frames = torch.from_numpy(edit_frames).permute(0, 3, 1, 2).contiguous() / 255.
                return torch.cat([orig_frames, ground_frames, edit_frames], dim=0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]

        while True:
            try:
                pixel_values = self.load_video_pair(
                    data_info['original_video'],
                    data_info.get('grounded_video', data_info.get('ground_video')),
                    data_info['edited_video'],
                )

                # Prepare instructions
                edit_text = data_info.get('edit_instruction', data_info.get('text', ''))
                ground_instr = derive_ground_object_from_instruction(edit_text)

                text = edit_text
                if self.instruction_template:
                    text = self.instruction_template.format(edit_instruction=edit_text, ground_instrction=ground_instr)

                if random.random() < self.text_drop_ratio:
                    text = ''

                sample = {
                    "pixel_values": pixel_values,
                    "text": text,
                    "data_type": "video",
                    "idx": idx,
                }

                if self.enable_inpaint and not self.enable_bucket:
                    pass

                return sample

            except Exception as e:
                print(f"Error loading video triplet: {e}")
                idx = random.randint(0, self.length-1)
    
class ImageVideoDataset(Dataset):
    def __init__(
        self,
        ann_path, data_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.0, 
        video_length_drop_end=1.0,
        enable_inpaint=False,
        return_file_name=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        # It's used to balance num of images and videos.
        if video_repeat > 0:
            self.dataset = []
            for data in dataset:
                if data.get('type', 'image') != 'video':
                    self.dataset.append(data)
                    
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        else:
            self.dataset = dataset
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.return_file_name = return_file_name

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        
        if data_info.get('type', 'image')=='video':
            video_id, text = data_info['file_path'], data_info['text']

            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)
                
                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''
            return pixel_values, text, 'video', video_dir
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)
            if random.random() < self.text_drop_ratio:
                text = ''
            return image, text, 'image', image_path

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, name, data_type, file_path = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(file_path)
                
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

class ImageVideoEditDataset(Dataset):
    def __init__(
        self,
        ann_path,
        data_root=None,
        video_sample_size=512,
        video_sample_stride=1,
        source_frames=33,
        target_frames=32,
        text_drop_ratio=0.1,
        enable_bucket=False,
        enable_inpaint=False,
        video_length_drop_start=0.0,
        video_length_drop_end=1.0,
        instruction_template="A video sequence showing two parts: the first half shows the original scene, and the second half shows the same scene but {edit_instruction}",
    ):
        dataset = json.load(open(ann_path))
        if isinstance(dataset, dict):
            new_dataset = []
            for _, info in dataset.items():
                # Keep original keys, just standardize text field
                data_type = info.get("type", "video")
                entry = dict(info)  # Copy original entry
                # Standardize text field name and handle None/empty values
                if "edit_instruction" in entry:
                    entry["text"] = entry["edit_instruction"]
                elif "instruction" in entry:
                    entry["text"] = entry["instruction"]
                elif "text" not in entry:
                    entry["text"] = ""
                
                # Ensure text is not None (convert None to empty string)
                if entry["text"] is None:
                    entry["text"] = ""
                
                # Add file_path for bucket sampler compatibility
                # Bucket sampler expects 'file_path' to get dimensions
                if data_type == "video":
                    entry["file_path"] = entry.get("original_video", "")
                else:  # image
                    entry["file_path"] = entry.get("original_image", "")
                
                new_dataset.append(entry)
            dataset = new_dataset

        self.data_root = data_root
        self.dataset = dataset
        self.length = len(self.dataset)

        # sampling params
        self.video_sample_stride = video_sample_stride
        self.source_frames = source_frames
        self.target_frames = target_frames
        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # transforms params (match ImageVideoDataset)
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        
        # Image transforms for non-bucket mode
        self.image_transforms = transforms.Compose([
            transforms.Resize(min(self.video_sample_size)),
            transforms.CenterCrop(self.video_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.instruction_template = instruction_template
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint

        # For pre-resize like ImageVideoDataset
        self.larger_side_of_image_and_video = min(self.video_sample_size)

    def _resize_and_center_crop_batch(self, frames_np, target_h, target_w):
        resized = []
        for i in range(frames_np.shape[0]):
            frame = frames_np[i]
            h, w = frame.shape[0], frame.shape[1]
            scale = max(target_h / h, target_w / w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            frame_resized = cv2.resize(frame, (new_w, new_h))
            y0 = max((new_h - target_h) // 2, 0)
            x0 = max((new_w - target_w) // 2, 0)
            frame_cropped = frame_resized[y0:y0 + target_h, x0:x0 + target_w]
            resized.append(frame_cropped)
        return np.stack(resized, axis=0)

    def _resize_and_center_crop_image(self, image_np, target_h, target_w):
        h, w = image_np.shape[0], image_np.shape[1]
        scale = max(target_h / h, target_w / w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        image_resized = cv2.resize(image_np, (new_w, new_h))
        y0 = max((new_h - target_h) // 2, 0)
        x0 = max((new_w - target_w) // 2, 0)
        image_cropped = image_resized[y0:y0 + target_h, x0:x0 + target_w]
        return image_cropped

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]

        data_type = data_info.get('type', 'video')
        
        # Handle None or empty instruction with safety fallback
        raw_text = data_info.get('text', '')
        if raw_text is None or (isinstance(raw_text, str) and not raw_text.strip()):
            # Use a generic fallback description if instruction is missing
            raw_text = "the content has been modified"
        
        # Apply instruction template if available
        if self.instruction_template and "{edit_instruction}" in self.instruction_template:
            text = self.instruction_template.format(edit_instruction=raw_text)
        else:
            text = raw_text

        if data_type == 'video':
            # video pair branch (default)
            src_rel, tgt_rel = data_info['original_video'], data_info['edited_video']

            if self.data_root is not None:
                src_path = os.path.join(self.data_root, src_rel)
                tgt_path = os.path.join(self.data_root, tgt_rel)
            else:
                src_path = src_rel
                tgt_path = tgt_rel

            # Force use CPU decoder to read all frames instead of just keyframes
            from decord import cpu
            with VideoReader_contextmanager(src_path, num_threads=2, ctx=cpu(0)) as src_reader, \
                 VideoReader_contextmanager(tgt_path, num_threads=2, ctx=cpu(0)) as tgt_reader:

                # Get video lengths
                src_length = len(src_reader)
                tgt_length = len(tgt_reader)
                
                # Check if video has enough frames
                if src_length < self.source_frames:
                    raise ValueError(f"Source video only has {src_length} frames, but requested {self.source_frames}")
                if tgt_length < self.target_frames:
                    raise ValueError(f"Target video only has {tgt_length} frames, but requested {self.target_frames}")

                # Unified sampling strategy: start from beginning (same as VideoEditDataset)
                start_idx = 0
                
                src_indices = np.linspace(
                    start_idx,
                    min(start_idx + (self.source_frames - 1) * self.video_sample_stride, src_length - 1),
                    self.source_frames,
                    dtype=int
                )
                
                tgt_indices = np.linspace(
                    start_idx,
                    min(start_idx + (self.target_frames - 1) * self.video_sample_stride, tgt_length - 1),
                    self.target_frames,
                    dtype=int
                )

                # read batches with timeout
                try:
                    src_frames = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=(src_reader, src_indices))
                    tgt_frames = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=(tgt_reader, tgt_indices))
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from pair. Error is {e}.")

                # align HxW between source and target to enable concat
                sh, sw = src_frames.shape[1], src_frames.shape[2]
                th, tw = tgt_frames.shape[1], tgt_frames.shape[2]
                target_h = min(sh, th)
                target_w = min(sw, tw)
                if (sh != target_h or sw != target_w):
                    src_frames = self._resize_and_center_crop_batch(src_frames, target_h, target_w)
                if (th != target_h or tw != target_w):
                    tgt_frames = self._resize_and_center_crop_batch(tgt_frames, target_h, target_w)

                if not self.enable_bucket:
                    src_tensor = torch.from_numpy(src_frames).permute(0, 3, 1, 2).contiguous() / 255.
                    tgt_tensor = torch.from_numpy(tgt_frames).permute(0, 3, 1, 2).contiguous() / 255.

                    src_tensor = self.video_transforms(src_tensor)
                    tgt_tensor = self.video_transforms(tgt_tensor)
                else:
                    src_tensor = src_frames
                    tgt_tensor = tgt_frames
                
                # Random text drop
                if random.random() < self.text_drop_ratio:
                    text = ''
                
            return src_tensor, tgt_tensor, text, 'video'
        else:
            # image pair branch (simple like ImageVideoDataset image path)
            src_img_rel = data_info.get('original_image')
            tgt_img_rel = data_info.get('edited_image')
            if src_img_rel is None or tgt_img_rel is None:
                raise ValueError('Missing original_image/edited_image for image sample')

            if self.data_root is not None:
                src_img_path = os.path.join(self.data_root, src_img_rel)
                tgt_img_path = os.path.join(self.data_root, tgt_img_rel)
            else:
                src_img_path = src_img_rel
                tgt_img_path = tgt_img_rel

            src_img = Image.open(src_img_path).convert('RGB')
            tgt_img = Image.open(tgt_img_path).convert('RGB')

            if not self.enable_bucket:
                # Apply transforms and add frame dimension
                src_tensor = self.image_transforms(src_img).unsqueeze(0)  # (1, C, H, W)
                tgt_tensor = self.image_transforms(tgt_img).unsqueeze(0)  # (1, C, H, W)
            else:
                # For bucket mode, keep as numpy and add frame dimension
                src_tensor = np.expand_dims(np.array(src_img), axis=0)  # (1, H, W, C)
                tgt_tensor = np.expand_dims(np.array(tgt_img), axis=0)  # (1, H, W, C)

            if random.random() < self.text_drop_ratio:
                text = ''
            
            return src_tensor, tgt_tensor, text, 'image'

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'video')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'video')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                src_vals, tgt_vals, name, data_type = self.get_batch(idx)
                if data_type == 'video':
                    sample["pixel_values_src_video"] = src_vals
                    sample["pixel_values_tgt_video"] = tgt_vals
                else:
                    sample["pixel_values_src_image"] = src_vals
                    sample["pixel_values_tgt_image"] = tgt_vals
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        # Inpaint not applied here to avoid ambiguity across src/tgt branches

        return sample


class ImageVideoCoTDataset(Dataset):
    """
    Dataset for Chain-of-Thought (CoT) style image/video editing.
    - For videos: loads original_video, grounded_video, and edited_video (3-part)
    - For images: loads original_image and edited_image (2-part, same as ImageVideoEditDataset)
    """
    def __init__(
        self,
        ann_path,
        data_root=None,
        video_sample_size=512,
        video_sample_stride=1,
        source_frames=33,
        reasoning_frames=4,
        target_frames=33,
        text_drop_ratio=0.1,
        enable_bucket=False,
        enable_inpaint=False,
        video_length_drop_start=0.0,
        video_length_drop_end=1.0,
        instruction_template="A video sequence showing three parts: first the original scene, then grounded {ground_instruction}, and finally the same scene but {edit_instruction}",
        enable_gradual_ground=False,
        enable_gray_red_mask=False,
        enable_gray_black_background=False,
        enable_gray_alpha_overlay=False,
        gray_alpha=0.5,
        gray_intensity_range=(96, 160),
        gray_tolerance=12,
    ):
        dataset = json.load(open(ann_path))
        if isinstance(dataset, dict):
            new_dataset = []
            for _, info in dataset.items():
                data_type = info.get("type", "video")
                entry = dict(info)  # Copy original entry
                
                # Standardize text field name and handle None/empty values
                if "edit_instruction" in entry:
                    entry["text"] = entry["edit_instruction"]
                elif "instruction" in entry:
                    entry["text"] = entry["instruction"]
                elif "text" not in entry:
                    entry["text"] = ""
                
                # Ensure text is not None
                if entry["text"] is None:
                    entry["text"] = ""
                
                # Add file_path for bucket sampler compatibility
                if data_type == "video":
                    entry["file_path"] = entry.get("original_video", "")
                else:  # image
                    entry["file_path"] = entry.get("original_image", "")
                
                new_dataset.append(entry)
            dataset = new_dataset

        self.data_root = data_root
        self.dataset = dataset
        self.length = len(self.dataset)

        # sampling params
        self.video_sample_stride = video_sample_stride
        self.source_frames = source_frames
        self.reasoning_frames = reasoning_frames
        self.target_frames = target_frames
        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # transforms params
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        
        # Image transforms for non-bucket mode
        self.image_transforms = transforms.Compose([
            transforms.Resize(min(self.video_sample_size)),
            transforms.CenterCrop(self.video_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.instruction_template = instruction_template
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.enable_gradual_ground = enable_gradual_ground
        # only one visualization mode at a time
        enabled_modes = int(bool(enable_gray_red_mask)) + int(bool(enable_gray_black_background)) + int(bool(enable_gray_alpha_overlay))
        if enabled_modes > 1:
            raise ValueError("enable_gray_red_mask, enable_gray_black_background and enable_gray_alpha_overlay cannot be enabled simultaneously.")
        self.enable_gray_red_mask = enable_gray_red_mask
        self.enable_gray_black_background = enable_gray_black_background
        self.enable_gray_alpha_overlay = enable_gray_alpha_overlay
        self.gray_alpha = float(gray_alpha)
        if not (0.0 <= self.gray_alpha <= 1.0):
            raise ValueError("gray_alpha must be in [0,1].")
        if not isinstance(gray_intensity_range, (list, tuple)) or len(gray_intensity_range) != 2:
            raise ValueError("gray_intensity_range must contain exactly two values (min and max intensity).")
        self.gray_intensity_range = (int(gray_intensity_range[0]), int(gray_intensity_range[1]))
        if self.gray_intensity_range[0] > self.gray_intensity_range[1]:
            raise ValueError("gray_intensity_range min value cannot be greater than max value.")
        self.gray_tolerance = int(gray_tolerance)

        # For pre-resize like ImageVideoDataset
        self.larger_side_of_image_and_video = min(self.video_sample_size)

    def _resize_and_center_crop_batch(self, frames_np, target_h, target_w):
        resized = []
        for i in range(frames_np.shape[0]):
            frame = frames_np[i]
            h, w = frame.shape[0], frame.shape[1]
            scale = max(target_h / h, target_w / w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            frame_resized = cv2.resize(frame, (new_w, new_h))
            y0 = max((new_h - target_h) // 2, 0)
            x0 = max((new_w - target_w) // 2, 0)
            frame_cropped = frame_resized[y0:y0 + target_h, x0:x0 + target_w]
            resized.append(frame_cropped)
        return np.stack(resized, axis=0)

    def _resize_and_center_crop_image(self, image_np, target_h, target_w):
        h, w = image_np.shape[0], image_np.shape[1]
        scale = max(target_h / h, target_w / w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        image_resized = cv2.resize(image_np, (new_w, new_h))
        y0 = max((new_h - target_h) // 2, 0)
        x0 = max((new_w - target_w) // 2, 0)
        image_cropped = image_resized[y0:y0 + target_h, x0:x0 + target_w]
        return image_cropped

    def _derive_ground_instruction(self, edit_instruction_text: str) -> str:
        """Derive grounded object phrase from instruction using shared rules."""
        return derive_ground_object_from_instruction(edit_instruction_text)

    def _ensure_same_size_pair(self, img_a: np.ndarray, img_b: np.ndarray) -> tuple:
        """Resize img_b to img_a's size if needed to enable per-pixel interpolation."""
        ha, wa = img_a.shape[:2]
        hb, wb = img_b.shape[:2]
        if (ha, wa) == (hb, wb):
            return img_a, img_b
        resized_b = cv2.resize(img_b, (wa, ha), interpolation=cv2.INTER_LINEAR)
        return img_a, resized_b

    def _interpolate_ground_frames(self, ground_first: np.ndarray, target_first: np.ndarray,
                                   total_steps: int = 16,
                                   pick_indices: tuple = (0, 4, 8, 12)) -> np.ndarray:
        """
        Create grounding frames by linearly interpolating between the first frame of
        the grounding video and the first frame of the edited video, then picking
        specific indices.
        Returns array of shape (len(pick_indices), H, W, 3) in uint8.
        """
        a_np, b_np = self._ensure_same_size_pair(ground_first, target_first)

        a_t = torch.from_numpy(a_np).float() / 255.0  # H, W, C
        b_t = torch.from_numpy(b_np).float() / 255.0  # H, W, C

        a_t = a_t.permute(2, 0, 1).contiguous()  # C, H, W
        b_t = b_t.permute(2, 0, 1).contiguous()  # C, H, W

        c, h, w = a_t.shape
        pair = torch.stack([a_t, b_t], dim=0)  # 2, C, H, W
        pair_chw_t = pair.permute(1, 2, 3, 0).contiguous()  # C, H, W, 2
        seq = pair_chw_t.view(1, c * h * w, 2)  # 1, (C*H*W), 2
        with torch.no_grad():
            seq_interp = F.interpolate(seq, size=int(total_steps), mode="linear", align_corners=True)
        seq_interp = seq_interp.view(c, h, w, int(total_steps)).permute(3, 0, 1, 2).contiguous()  # T, C, H, W

        out_frames = []
        t_steps = int(total_steps)
        for idx in pick_indices:
            safe_idx = max(0, min(int(idx), t_steps - 1))
            img = (seq_interp[safe_idx].clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
            out_frames.append(img)
        return np.stack(out_frames, axis=0)

    def _build_gray_mask(self, frame: np.ndarray) -> np.ndarray:
        """Detect gray regions in a frame using intensity range and tolerance."""
        frame_float = frame.astype(np.float32)
        if frame_float.max() <= 1.0:
            frame_float = frame_float * 255.0
        channel_max = frame_float.max(axis=2)
        channel_min = frame_float.min(axis=2)
        min_intensity, max_intensity = self.gray_intensity_range
        tone_flatness = channel_max - channel_min
        mask = tone_flatness <= float(self.gray_tolerance)
        mask &= channel_max >= float(min_intensity)
        mask &= channel_max <= float(max_intensity)
        return mask

    def _apply_gray_region_effect(self, frames_np: np.ndarray, mode: str) -> np.ndarray:
        """Apply requested effect on detected gray regions for a batch of frames."""
        processed_frames = []
        for frame in frames_np:
            mask = self._build_gray_mask(frame)
            if not np.any(mask):
                processed_frames.append(frame)
                continue
            frame_out = frame.copy()
            if np.issubdtype(frame_out.dtype, np.floating) and frame_out.max() <= 1.0:
                red_value = np.array([1.0, 0.0, 0.0], dtype=frame_out.dtype)
            else:
                red_value = np.array([255, 0, 0], dtype=frame_out.dtype)
            if mode == "red":
                frame_out[mask] = red_value
            else:
                frame_out[:] = 0
                frame_out[mask] = frame[mask]
            processed_frames.append(frame_out)
        return np.stack(processed_frames, axis=0)

    def _apply_gray_overlay_from_reference(self, src_frames_np: np.ndarray, ref_frames_np: np.ndarray,
                                           alpha: float = 0.5, gray_value: float = 0.5, num_frames: int = 4) -> np.ndarray:
        """
        Detect gray regions on ref frames, and overlay gray with alpha onto the
        first `num_frames` frames of src frames at the same positions.
        """
        n = min(int(num_frames), int(src_frames_np.shape[0]), int(ref_frames_np.shape[0]))
        if n <= 0:
            return src_frames_np
        out = src_frames_np.copy()
        a = float(alpha)
        a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
        gv = float(gray_value)
        gv = 0.0 if gv < 0.0 else (1.0 if gv > 1.0 else gv)
        for i in range(n):
            mask = self._build_gray_mask(ref_frames_np[i])
            if not np.any(mask):
                continue
            src = out[i]
            # normalize to 0..1 float
            if np.issubdtype(src.dtype, np.floating):
                f = src.astype(np.float32)
                if f.max() > 1.0:
                    f = np.clip(f / 255.0, 0.0, 1.0)
                back_to_uint8 = False
            else:
                f = src.astype(np.float32) / 255.0
                back_to_uint8 = True
            gray_color = np.array([gv, gv, gv], dtype=np.float32)
            # boolean mask is (H,W); f[mask] -> (K,3), broadcast with gray_color (3,)
            f[mask] = (1.0 - a) * f[mask] + a * gray_color
            if back_to_uint8:
                out[i] = (f * 255.0).clip(0, 255).astype(src.dtype)
            else:
                out[i] = f.astype(src.dtype)
        return out

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'video')
        
        # Handle None or empty instruction with safety fallback
        raw_text = data_info.get('text', '')
        if raw_text is None or (isinstance(raw_text, str) and not raw_text.strip()):
            raw_text = "the content has been modified"
        
        if data_type == 'video':
            # Video triplet branch: original + grounded + edited
            src_rel = data_info['original_video']
            # Support both 'grounded_video' and 'ground_video' keys
            ground_rel = data_info.get('grounded_video', data_info.get('ground_video'))
            tgt_rel = data_info['edited_video']

            if self.data_root is not None:
                src_path = os.path.join(self.data_root, src_rel)
                ground_path = os.path.join(self.data_root, ground_rel)
                tgt_path = os.path.join(self.data_root, tgt_rel)
            else:
                src_path = src_rel
                ground_path = ground_rel
                tgt_path = tgt_rel

            # Force use CPU decoder to read all frames
            from decord import cpu
            with VideoReader_contextmanager(src_path, num_threads=2, ctx=cpu(0)) as src_reader, \
                 VideoReader_contextmanager(ground_path, num_threads=2, ctx=cpu(0)) as ground_reader, \
                 VideoReader_contextmanager(tgt_path, num_threads=2, ctx=cpu(0)) as tgt_reader:

                # Get video lengths
                src_length = len(src_reader)
                ground_length = len(ground_reader)
                tgt_length = len(tgt_reader)
                
                # Check if video has enough frames
                if src_length < self.source_frames:
                    raise ValueError(f"Source video only has {src_length} frames, but requested {self.source_frames}")
                if tgt_length < self.target_frames:
                    raise ValueError(f"Target video only has {tgt_length} frames, but requested {self.target_frames}")

                # Unified sampling strategy: start from beginning
                start_idx = 0
                
                # Sample source frames
                src_indices = np.linspace(
                    start_idx,
                    min(start_idx + (self.source_frames - 1) * self.video_sample_stride, src_length - 1),
                    self.source_frames,
                    dtype=int
                )
                
                # Sample target frames
                tgt_indices = np.linspace(
                    start_idx,
                    min(start_idx + (self.target_frames - 1) * self.video_sample_stride, tgt_length - 1),
                    self.target_frames,
                    dtype=int
                )

                # Read batches with timeout
                try:
                    src_frames = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=(src_reader, src_indices))
                    tgt_frames = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=(tgt_reader, tgt_indices))

                    if self.enable_gradual_ground:
                        # Interpolate between first frame of grounded and edited videos
                        ground_first = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=(ground_reader, [0]))
                        # Use the first decoded edited frame if available to avoid double decode
                        tgt_first_frame = tgt_frames[0]
                        # steps: 0..15, pick 0,3,6,9,12 -> 5 grounding frames
                        ground_frames = self._interpolate_ground_frames(
                            ground_first=ground_first[0],
                            target_first=tgt_first_frame,
                            total_steps=16,
                            pick_indices=(0, 3, 6, 9, 12),
                        )
                    else:
                        # # Original behavior: sample grounding frames evenly by stride
                        # ground_indices = np.linspace(
                        #     start_idx,
                        #     min(start_idx + (self.reasoning_frames - 1) * self.video_sample_stride, ground_length - 1),
                        #     self.reasoning_frames,
                        #     dtype=int
                        # )
                        
                        #==============================================================
                        # New behavior: ground_indices are the first 'reasoning_frames' from src_indices
                        ground_indices = src_indices[:self.reasoning_frames]

                        # --- 增加这个重要的安全检查 ---
                        # 确保我们想采样的最后一帧 (ground_indices[-1])
                        # 没有超出 ground_video 的总长度 (ground_length)
                        if len(ground_indices) > 0 and ground_indices[-1] >= ground_length:
                            raise ValueError(
                                f"Data inconsistency error: Ground video has only {ground_length} frames, "
                                f"but the source-based sampling (stride={self.video_sample_stride}) "
                                f"requires reading up to frame {ground_indices[-1]}. "
                                f"File: {ground_path}"
                            )
                        ground_frames = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=(ground_reader, ground_indices))
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from triplet. Error is {e}.")

                # Align HxW among source, ground, and target to enable concat
                sh, sw = src_frames.shape[1], src_frames.shape[2]
                gh, gw = ground_frames.shape[1], ground_frames.shape[2]
                th, tw = tgt_frames.shape[1], tgt_frames.shape[2]
                target_h = min(sh, gh, th)
                target_w = min(sw, gw, tw)
                
                if (sh != target_h or sw != target_w):
                    src_frames = self._resize_and_center_crop_batch(src_frames, target_h, target_w)
                if (gh != target_h or gw != target_w):
                    ground_frames = self._resize_and_center_crop_batch(ground_frames, target_h, target_w)
                if (th != target_h or tw != target_w):
                    tgt_frames = self._resize_and_center_crop_batch(tgt_frames, target_h, target_w)

                if self.enable_gray_red_mask or self.enable_gray_black_background:
                    effect_mode = "red" if self.enable_gray_red_mask else "black"
                    ground_frames = self._apply_gray_region_effect(ground_frames, effect_mode)
                elif self.enable_gray_alpha_overlay:
                    # Use gray regions detected on grounding frames to overlay 50% gray on the
                    # first 4 frames of the original video.
                    ground_frames = self._apply_gray_overlay_from_reference(
                        src_frames, ground_frames, alpha=self.gray_alpha, gray_value=0.5, num_frames=4
                    )

                if not self.enable_bucket:
                    src_tensor = torch.from_numpy(src_frames).permute(0, 3, 1, 2).contiguous() / 255.
                    ground_tensor = torch.from_numpy(ground_frames).permute(0, 3, 1, 2).contiguous() / 255.
                    tgt_tensor = torch.from_numpy(tgt_frames).permute(0, 3, 1, 2).contiguous() / 255.

                    src_tensor = self.video_transforms(src_tensor)
                    ground_tensor = self.video_transforms(ground_tensor)
                    tgt_tensor = self.video_transforms(tgt_tensor)
                else:
                    src_tensor = src_frames
                    ground_tensor = ground_frames
                    tgt_tensor = tgt_frames
                # Prepare text with template
                ground_instr = self._derive_ground_instruction(raw_text)
                if self.instruction_template and "{edit_instruction}" in self.instruction_template:
                    text = self.instruction_template.format(
                        edit_instruction=raw_text,
                        ground_instruction=ground_instr
                    )
                else:    
                    text = raw_text
                
                # Random text drop
                if random.random() < self.text_drop_ratio:
                    text = ''
                
            return src_tensor, ground_tensor, tgt_tensor, text, 'video'
            
        else:
            # Image pair branch (simple like ImageVideoEditDataset)
            src_img_rel = data_info.get('original_image')
            tgt_img_rel = data_info.get('edited_image')
            if src_img_rel is None or tgt_img_rel is None:
                raise ValueError('Missing original_image/edited_image for image sample')

            if self.data_root is not None:
                src_img_path = os.path.join(self.data_root, src_img_rel)
                tgt_img_path = os.path.join(self.data_root, tgt_img_rel)
            else:
                src_img_path = src_img_rel
                tgt_img_path = tgt_img_rel

            src_img = Image.open(src_img_path).convert('RGB')
            tgt_img = Image.open(tgt_img_path).convert('RGB')

            if not self.enable_bucket:
                # Apply transforms and add frame dimension
                src_tensor = self.image_transforms(src_img).unsqueeze(0)  # (1, C, H, W)
                tgt_tensor = self.image_transforms(tgt_img).unsqueeze(0)  # (1, C, H, W)
            else:
                # For bucket mode, keep as numpy and add frame dimension
                src_tensor = np.expand_dims(np.array(src_img), axis=0)  # (1, H, W, C)
                tgt_tensor = np.expand_dims(np.array(tgt_img), axis=0)  # (1, H, W, C)

            # Apply instruction template if available
            if self.instruction_template and "{edit_instruction}" in self.instruction_template:
                text = self.instruction_template.format(edit_instruction=raw_text, ground_instruction="")
            else:
                text = raw_text

            if random.random() < self.text_drop_ratio:
                text = ''
            
            # For images, ground_tensor is None
            return src_tensor, None, tgt_tensor, text, 'image'

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'video')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'video')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                result = self.get_batch(idx)
                
                if data_type == 'video':
                    src_vals, ground_vals, tgt_vals, name, data_type = result
                    sample["pixel_values_src_video"] = src_vals
                    sample["pixel_values_ground_video"] = ground_vals
                    sample["pixel_values_tgt_video"] = tgt_vals
                else:
                    src_vals, _, tgt_vals, name, data_type = result
                    sample["pixel_values_src_image"] = src_vals
                    sample["pixel_values_tgt_image"] = tgt_vals
                    
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        return sample

def padding_image(images, new_width, new_height):
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    aspect_ratio = images.width / images.height
    if new_width / new_height > 1:
        if aspect_ratio > new_width / new_height:
            new_img_width = new_width
            new_img_height = int(new_img_width / aspect_ratio)
        else:
            new_img_height = new_height
            new_img_width = int(new_img_height * aspect_ratio)
    else:
        if aspect_ratio > new_width / new_height:
            new_img_width = new_width
            new_img_height = int(new_img_width / aspect_ratio)
        else:
            new_img_height = new_height
            new_img_width = int(new_img_height * aspect_ratio)

    resized_img = images.resize((new_img_width, new_img_height))

    paste_x = (new_width - new_img_width) // 2
    paste_y = (new_height - new_img_height) // 2

    new_image.paste(resized_img, (paste_x, paste_y))

    return new_image

class ImageVideoControlDataset(Dataset):
    def __init__(
        self,
        ann_path, data_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.1, 
        video_length_drop_end=0.9,
        enable_inpaint=False,
        enable_camera_info=False,
    ):
        # Loading annotations from files
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        # It's used to balance num of images and videos.
        if video_repeat > 0:
            self.dataset = []
            for data in dataset:
                if data.get('type', 'image') != 'video':
                    self.dataset.append(data)
                    
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        else:
            self.dataset = dataset
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint  = enable_inpaint
        self.enable_camera_info = enable_camera_info

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        if self.enable_camera_info:
            self.video_transforms_camera = transforms.Compose(
                [
                    transforms.Resize(min(self.video_sample_size)),
                    transforms.CenterCrop(self.video_sample_size)
                ]
            )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))
    
    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        video_id, text = data_info['file_path'], data_info['text']

        if data_info.get('type', 'image')=='video':
            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)
                
                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''

            control_video_id = data_info['control_file_path']

            if self.data_root is None:
                control_video_id = control_video_id
            else:
                control_video_id = os.path.join(self.data_root, control_video_id)
            
            if self.enable_camera_info:
                if control_video_id.lower().endswith('.txt'):
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)

                        control_camera_values = process_pose_file(control_video_id, width=self.video_sample_size[1], height=self.video_sample_size[0])
                        control_camera_values = torch.from_numpy(control_camera_values).permute(0, 3, 1, 2).contiguous()
                        control_camera_values = F.interpolate(control_camera_values, size=(len(video_reader), control_camera_values.size(3)), mode='bilinear', align_corners=True)
                        control_camera_values = self.video_transforms_camera(control_camera_values)
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)

                        control_camera_values = process_pose_file(control_video_id, width=self.video_sample_size[1], height=self.video_sample_size[0], return_poses=True)
                        control_camera_values = torch.from_numpy(np.array(control_camera_values)).unsqueeze(0).unsqueeze(0)
                        control_camera_values = F.interpolate(control_camera_values, size=(len(video_reader), control_camera_values.size(3)), mode='bilinear', align_corners=True)[0][0]
                        control_camera_values = np.array([control_camera_values[index] for index in batch_index])
                else:
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)
                        control_camera_values = None
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)
                        control_camera_values = None
            else:
                with VideoReader_contextmanager(control_video_id, num_threads=2) as control_video_reader:
                    try:
                        sample_args = (control_video_reader, batch_index)
                        control_pixel_values = func_timeout(
                            VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                        )
                        resized_frames = []
                        for i in range(len(control_pixel_values)):
                            frame = control_pixel_values[i]
                            resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                            resized_frames.append(resized_frame)
                        control_pixel_values = np.array(resized_frames)
                    except FunctionTimedOut:
                        raise ValueError(f"Read {idx} timeout.")
                    except Exception as e:
                        raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                    if not self.enable_bucket:
                        control_pixel_values = torch.from_numpy(control_pixel_values).permute(0, 3, 1, 2).contiguous()
                        control_pixel_values = control_pixel_values / 255.
                        del control_video_reader
                    else:
                        control_pixel_values = control_pixel_values

                    if not self.enable_bucket:
                        control_pixel_values = self.video_transforms(control_pixel_values)
                control_camera_values = None

            return pixel_values, control_pixel_values, control_camera_values, text, "video"
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)

            if random.random() < self.text_drop_ratio:
                text = ''

            control_image_id = data_info['control_file_path']

            if self.image_root is None:
                control_image_id = control_image_id
            else:
                control_image_id = os.path.join(self.image_root, control_image_id)

            control_image = Image.open(control_image_id).convert('RGB')
            if not self.enable_bucket:
                control_image = self.image_transforms(control_image).unsqueeze(0)
            else:
                control_image = np.expand_dims(np.array(control_image), 0)
            return image, control_image, None, text, 'image'
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, control_pixel_values, control_camera_values, name, data_type = self.get_batch(idx)

                sample["pixel_values"] = pixel_values
                sample["control_pixel_values"] = control_pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx

                if self.enable_camera_info:
                    sample["control_camera_values"] = control_camera_values

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.zeros_like(pixel_values) * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample