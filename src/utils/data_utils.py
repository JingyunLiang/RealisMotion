# Copyright 2025 The RealisMotion Team. All rights reserved.
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
import torch
import cv2
import numpy as np
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import decord
from PIL import Image
import json
import glob
from scipy.ndimage import binary_dilation
from functools import partial

decord.bridge.set_bridge("torch")


def trim(x, divide_by=8):
    if x % divide_by != 0:
        x = (x // divide_by ) * divide_by
    return x


def resize_long_and_pad_short_edge(img, width, height, interpolation=transforms.InterpolationMode.BILINEAR):
    # resie long side
    img_W, img_H = img.size
    scale = min(width / img_W, height / img_H)
    new_W, new_H = int(img_W * scale), int(img_H * scale)
    img = F.resize(img, (new_H, new_W), interpolation=interpolation)
    
    # pad short side
    img_W, img_H = img.size
    padding_left = (width - img_W) // 2
    padding_right = width - img_W - padding_left
    padding_top = (height - img_H) // 2
    padding_bottom = height - img_H - padding_top
    img = F.pad(img, (padding_left, padding_top, padding_right, padding_bottom), 0, "constant")
    
    return img


def dilate_mask(mask, kernel_size=55):
    channel_num = mask.shape[1]
    mask = mask[:, :1].cuda() # faster with one channel
    
    if kernel_size > 0:
        pad = (kernel_size - 1) // 2
        mask = torch.nn.functional.pad(mask, (pad, pad, pad, pad, 0, 0), mode='constant', value=0)
        mask = torch.nn.functional.max_pool3d(mask, kernel_size=(1, kernel_size, kernel_size), stride=1)
        mask = (mask > 0).float()

    mask = mask.repeat(1, channel_num, 1, 1, 1).cpu()

    return mask


def get_smpl_mask(image, kernel_size=55):
    mask = (((image.sum(dim=1, keepdim=True) + 1) / 2) > 0).float()
    
    # dilation
    mask = dilate_mask(mask, kernel_size=kernel_size)
    mask = mask.repeat(1, 3, 1, 1, 1)

    return mask


def is_image(path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    return path.lower().endswith(valid_extensions)


def load_meta_dicts(path):
    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(path, '*.jsonl')))

    meta_dicts = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            partial_json = ""
            for line in f:
                line = line.strip()
                if not line:
                    continue
                partial_json += line
                try:
                    json_obj = json.loads(partial_json)
                    meta_dicts.append(json_obj)
                    partial_json = ""
                except json.JSONDecodeError:
                    # JSON object not complete yet, continue parsing
                    partial_json += " "

    return meta_dicts


def load_image(path, div=127.5, add=-1):
    # read image
    image = torch.from_numpy(np.array(Image.open(path).convert("RGB")))
    # [0, 255] -> [-1, 1]
    image = image / div + add
    # H W C -> B C F H W
    image = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    return image


def load_video(
    path,
    fps=16,
    start_index=0,
    num_frames=97,
    div=127.5,
    add=-1,
):
    video_reader = decord.VideoReader(path)
    ori_fps = video_reader.get_avg_fps()
    ori_video_length = len(video_reader)
    if fps is None:
        fps = ori_fps
    
    clip_length = min(ori_video_length, round(num_frames / fps * ori_fps))
    start_index = min(start_index, ori_video_length - clip_length)
    clip_idxes = np.linspace(start_index, start_index + clip_length - 1, num_frames, dtype=int).tolist()
    pad = (4 - ((num_frames - 1) % 4)) %  4
    if pad != 0:
        clip_idxes = clip_idxes + clip_idxes[-1:] * pad

    video = video_reader.get_batch(clip_idxes).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    del video_reader
    video = video / div + add
    return video  # 1 C T H W


def load(path, fps, start_index, num_frames, div=127.5, add=-1):
    if is_image(path):
        result = load_image(path, div=div, add=add)
    else:
        result = load_video(path, fps=fps, start_index=start_index, num_frames=num_frames, div=div, add=add)
    return result


def detect_face(img_pil, dilate_ratio=0.5):
    height, width = img_pil.size[1], img_pil.size[0]
    face_detector = cv2.FaceDetectorYN_create('./src/utils/face_detection_yunet_2023mar.onnx', "", 
            input_size=(width, height), score_threshold=0.8, top_k=1)

    _, faces = face_detector.detect(cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))
    if faces is not None:
        x, y, w, h = faces[0, :4].astype(int) # we detect face on the masked image, so there are only one face.
        x = max(0, x - int(w * dilate_ratio / 2))
        y = max(0, y - int(h * dilate_ratio / 2))
        w = int(w * (1 + dilate_ratio))
        h = int(h * (1 + dilate_ratio))
        face = img_pil.crop((x, y, x + w, y + h))
    else:
        face = Image.new('RGB', (384, 384), (0, 0, 0))

    return face
  

def crop_and_detect_face(ref_path, ref_cs_map_path, ref_mask_path, fps, ref_index):
    ref_image = load(ref_path, fps=fps, start_index=ref_index, num_frames=1)

    # ref_cs_map and ref_mask are optional
    if ref_cs_map_path is not None:
        ref_cs_map = load(ref_cs_map_path, fps=fps, start_index=ref_index, num_frames=1)
        if ref_mask_path is not None and os.path.exists(ref_mask_path):
            ref_mask = load(ref_mask_path, fps=fps, start_index=ref_index, num_frames=1, div=255.0, add=0)
        else:
            ref_mask = get_smpl_mask(ref_cs_map, kernel_size=0)
    else:
        ref_cs_map = -torch.ones_like(ref_image)
        if ref_mask_path is not None and os.path.exists(ref_mask_path):
            ref_mask = load(ref_mask_path, fps=fps, start_index=ref_index, num_frames=1, div=255.0, add=0)
        else:
            ref_mask = torch.ones_like(ref_image)

    # tensor to pil
    ref_image_pil = Image.fromarray((((ref_image + 1) / 2).squeeze().permute(1, 2, 0) * 255.).byte().numpy()).convert("RGB")
    ref_cs_map_pil = Image.fromarray((((ref_cs_map + 1) / 2).squeeze().permute(1, 2, 0) * 255.).byte().numpy()).convert("RGB")
    ref_mask_pil = Image.fromarray((ref_mask.squeeze().permute(1, 2, 0) * 255.).byte().numpy()).convert("RGB")
    
    # crop the target human first if the bbx is available
    bbx_path = os.path.join(os.path.dirname(ref_path), 'preprocess/bbx.pt')
    if os.path.exists(bbx_path):
        x_min, y_min, x_max, y_max = torch.load(bbx_path, weights_only=True)['bbx_xyxy'][ref_index]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    else:
        x_min, y_min, x_max, y_max = 0, 0, ref_image_pil.size[0], ref_image_pil.size[1]
    x_max = x_min + trim(x_max - x_min)
    y_max = y_min + trim(y_max - y_min)

    ref_image_pil = ref_image_pil.crop((x_min, y_min, x_max, y_max))
    ref_cs_map_pil = ref_cs_map_pil.crop((x_min, y_min, x_max, y_max))
    ref_mask_pil = ref_mask_pil.crop((x_min, y_min, x_max, y_max))

    # # mask the ref img
    # ref_image_array = np.array(ref_image_pil)
    # ref_mask_array = np.array(ref_mask_pil.convert("L"))
    # ref_image_array = np.where(ref_mask_array[:, :, None] > 0, ref_image_array, 0)
    # ref_image_pil = Image.fromarray(ref_image_array)

    # transform
    ref_image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5], inplace=True),
        transforms.Lambda(lambda x: x.unsqueeze(0).unsqueeze(2)),
    ])
    ref_image = ref_image_transform(ref_image_pil)
    ref_cs_map = ref_image_transform(ref_cs_map_pil)
    ref_mask = ref_image_transform(ref_mask_pil)

    # mask the ref img (this is better than using PIL.Image, which leads to contour problems.)
    ref_image = (((ref_image + 1) / 2) * ((ref_mask + 1) / 2)) / 0.5 - 1
    ref_image_pil = Image.fromarray((((ref_image + 1) / 2).squeeze().permute(1, 2, 0) * 255.).byte().numpy()).convert("RGB")

    # face
    ref_face_pil = detect_face(ref_image_pil)
    face_transform = transforms.Compose([
        transforms.Lambda(partial(resize_long_and_pad_short_edge, width=384, height=384)),
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5], inplace=True),
        transforms.Lambda(lambda x: x.unsqueeze(0).unsqueeze(2)),
    ])
    ref_face = face_transform(ref_face_pil)

    return ref_image, ref_cs_map, ref_face
    