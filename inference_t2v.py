import argparse
import numpy as np
import os
import cv2
import torch
import tqdm

from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from src.pipelines.rm_dit_pipeline import RealisMotionPipeline
from src.utils.dist_utils import hook_for_multi_gpu_inference, init_dist, is_main_process, set_seed
from src.utils.data_utils import load, crop_and_detect_face, load_meta_dicts, get_smpl_mask, dilate_mask

FPS = 16

def inference(pipe, prompt, 
                ref_path, ref_cs_map_path, ref_mask_path, ref_index,
                background_path, background_mask_path, background_start_index, background_num_frames,
                motion_folder, motion_start_index, motion_num_frames, use_cfg_on_text,
                save_dir, enable_teacache):
    assert background_num_frames == motion_num_frames or background_num_frames == 1

    if prompt is None:
        prompt = "一个人在做某个动作，高质量，高分辨率，真实风格，细节真实自然，细节清晰，色调自然，光影自然，美丽的，漂亮的，形态自然，姿态自然，自然的动作，自然的表情，脸部清晰，清楚的，穿着鞋子，穿着衣服，干净的背景，稳定的背景， 自然的手指，紧闭的嘴巴"
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，手指不自然，手指乱动，说话，嘴巴乱动，眼神不自然"

    # crop the human and face out
    ref_image, ref_cs_map, ref_face = crop_and_detect_face(ref_path, ref_cs_map_path, ref_mask_path, fps=None, ref_index=ref_index)
    # mask the background
    ref_list = [ref_image, ref_cs_map, ref_face]
    
    # load background and mask
    background = load(background_path, fps=FPS, start_index=background_start_index, num_frames=background_num_frames)

    # background mask is optional
    if background_mask_path is not None and os.path.exists(background_mask_path):
        background_mask = load(background_mask_path, fps=FPS, start_index=background_start_index, num_frames=background_num_frames, div=255.0, add=0)
        background_mask = dilate_mask(background_mask, kernel_size=5)
    else:
        # get the mask from the original smpl
        background_motion_depth = os.path.join(os.path.dirname(background_path), 'depth_map.mp4')
        if os.path.exists(background_motion_depth):
            background_motion_depth = load(background_motion_depth, fps=FPS, start_index=background_start_index, num_frames=background_num_frames)
            background_mask = get_smpl_mask(background_motion_depth, kernel_size=55)
        else:
            background_mask = torch.zeros_like(background)
    
    # repeat the still background
    if background.shape[2] == 1:
        background = background.repeat(1, 1, motion_num_frames, 1, 1)
        background_mask = background_mask.repeat(1, 1, motion_num_frames, 1, 1)

    # load motion
    motion_list = []
    for motion in ['depth_map.mp4', 'normal_map.mp4', 'cs_map.mp4', 'hamer.mp4']:
        motion_path = os.path.join(motion_folder, motion)
        if os.path.exists(motion_path):
            motion_list.append(load(motion_path, fps=FPS, start_index=motion_start_index, num_frames=motion_num_frames))
        else:
            # hamer.mp4 is optional
            motion_list.append(torch.zeros_like(motion_list[0]))

    # merge the original background mask and the motion mask
    motion_mask = get_smpl_mask(motion_list[0], kernel_size=55)
    mask = ((background_mask + motion_mask)[:, :1] > 0).float()
    background = (((background + 1) / 2) * (1 - mask)) / 0.5 - 1
    mask = mask / 0.5 - 1

    # inference pipeline
    output = pipe(
        ref_list=ref_list,
        background=background,
        mask=mask,
        motion_list=motion_list,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=background.shape[2],
        enable_teacache=enable_teacache,
        use_cfg_on_text=use_cfg_on_text,
    ).frames[0]

    output_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(background_path))[0]}_{os.path.splitext(os.path.basename(motion_folder))[0]}_{os.path.splitext(os.path.basename(ref_path))[0]}.mp4")
                                               
    if is_main_process():
        export_to_video(output, output_path, fps=FPS)


def main():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path', type=str, default=None, help='path to reference image or video.')
    parser.add_argument('--ref_cs_map_path', type=str, default=None, help='path to reference cs_map, optional.')
    parser.add_argument('--ref_mask_path', type=str, default=None, help='path to reference image mask, optional.')
    parser.add_argument('--background_path', type=str, default=None, help='path to background image or video.')
    parser.add_argument('--background_mask_path', type=str, default=None, help='path to background image or video mask, optional.')
    parser.add_argument('--motion_folder', type=str, default=None, help='Path to motion folder path, with depth_map.mp4, normal_map.mp4, cs_map.mp4 and hamer.mp4 (optional).')
    parser.add_argument('--ref_index', type=int, default=0, help='index for ref video frame, optional.')
    parser.add_argument('--background_start_index', type=int, default=0, help='start index for background video frame.')
    parser.add_argument('--background_num_frames', type=int, default=1, help='num of background video frames. 1 for static background.')
    parser.add_argument('--motion_start_index', type=int, default=0, help='start index for motion video frame.')
    parser.add_argument('--motion_num_frames', type=int, default=97, help='num of motion video frames. Equals to the resulting video.')
    parser.add_argument('--use_cfg_on_text', type=bool, default=False, help='use use_cfg_on_text will result in worse identity preservation abiltiy from ref images.')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt for video.')
    parser.add_argument('--jsonl_path', type=str, default=None, help='Jsonl path for batch inference.')
    parser.add_argument('--save-dir', type=str, default="./output", help='Path to output folder.')
    parser.add_argument('--ckpt', type=str, default="./pretrained_models/RealisMotion", help='Path to checkpoint folder.')
    parser.add_argument('--seed', type=int, default=1024, help='The generation seed.')
    parser.add_argument('--save-gpu-memory', action='store_true', help='Save GPU memory, but will be super slow.')
    parser.add_argument(
        '--multi-gpu', action='store_true', help='Enable FSDP and Sequential parallel for multi-GPU inference.',
    )
    parser.add_argument(
        '--enable-teacache', action='store_true',
        help='Enable teacache to accelerate inference. Note that enabling teacache may hurt generation quality.',
    )
    args = parser.parse_args()

    # assign args
    ref_path = args.ref_path
    ref_cs_map_path = args.ref_cs_map_path
    ref_mask_path = args.ref_mask_path
    background_path = args.background_path
    background_mask_path = args.background_mask_path
    motion_folder = args.motion_folder
    ref_index = args.ref_index
    background_start_index = args.background_start_index
    background_num_frames = args.background_num_frames
    motion_start_index = args.motion_start_index
    motion_num_frames = args.motion_num_frames
    use_cfg_on_text = args.use_cfg_on_text
    prompt = args.prompt
    jsonl_path = args.jsonl_path
    save_dir = args.save_dir
    ckpt = args.ckpt
    seed = args.seed
    save_gpu_memory = args.save_gpu_memory
    multi_gpu = args.multi_gpu
    enable_teacache = args.enable_teacache
    os.makedirs(save_dir, exist_ok=True)

    # check args
    if save_gpu_memory and multi_gpu:
        raise ValueError("`--multi-gpu` and `--save-gpu-memory` cannot be set at the same time.")

    # init dist and set seed
    if multi_gpu:
        init_dist()
    set_seed(seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # load model
    model_id = ckpt
    pipe = RealisMotionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    if save_gpu_memory:
        print("WARNING: Enable sequential cpu offload which will be super slow.")
        pipe.enable_sequential_cpu_offload()
    elif multi_gpu:
        pipe = hook_for_multi_gpu_inference(pipe)
    else:
        pipe.enable_model_cpu_offload()

    # inference
    if jsonl_path is not None:  # batch inference
        for meta_dict in tqdm(load_meta_dicts(jsonl_path)):
            ref_path = meta_dict['meta_info_reference']['path']
            ref_cs_map_path = os.path.join(meta_dict['analysis_result_reference']['video_smpl_gvhmr'][0]['results_path'], 'cs_map.mp4')
            ref_mask_path = os.path.join(meta_dict['analysis_result_reference']['video_smpl_gvhmr'][0]['results_path'], 'mask.mp4')
            background_path = meta_dict['meta_info_background']['path']
            background_mask_path = os.path.join(meta_dict['analysis_result_reference']['video_smpl_gvhmr'][0]['results_path'], 'mask.mp4')
            motion_folder = meta_dict['analysis_result_reference']['video_smpl_gvhmr'][0]['results_path']
            ref_index = meta_dict['meta_info_reference']['start_frame']
            background_start_index = meta_dict['meta_info_background']['start_frame']
            if meta_dict['meta_info_background']['static_video'] == 1:
                background_num_frames = 1
            else:
                background_num_frames = args.motion_num_frames
            motion_start_index = meta_dict['meta_info_motion']['start_frame']
            motion_num_frames = args.motion_num_frames
            use_cfg_on_text = args.use_cfg_on_text
            prompt = meta_dict['analysis_result_background']['caption'][0]['text']
            inference(pipe, prompt, 
                ref_path, ref_cs_map_path, ref_mask_path, ref_index,
                background_path, background_mask_path, background_start_index, background_num_frames,
                motion_folder, motion_start_index, motion_num_frames, use_cfg_on_text.
                save_dir, enable_teacache)
    
    else:  # single sample inference
        inference(pipe, prompt, 
                ref_path, ref_cs_map_path, ref_mask_path, ref_index,
                background_path, background_mask_path, background_start_index, background_num_frames,
                motion_folder, motion_start_index, motion_num_frames, use_cfg_on_text,
                save_dir, enable_teacache)


if __name__ == "__main__":
    main()
