import os
import cv2
import shutil
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_module, compose
from omegaconf import OmegaConf 
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle, axis_angle_to_quaternion, quaternion_to_axis_angle

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    save_video,
    merge_videos_horizontal,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch

from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SLAMModel

from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from tqdm import tqdm
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from einops import einsum, rearrange
from PIL import Image

import torch
import pickle
import roma
import copy
from hmc.render_demo import run_preprocess, load_data_dict
from hmc.utils.renderer_depth_normal_cs_map_hamer import Renderer as ConditionRenderer
from hmc.utils.video_io_utils import get_video_fps
from hmc.utils.edit_utils import clip_and_repeat_smpl, repeat_smpl_at_begin_end, move_to_ground, get_rotation_matrix_xz_plane, move_to_ground_transl, transform_smpl_params_incam_from_global, world2incam, incam2world, get_trajectory2d, get_bbox_from_vertices, interpolate_curve, get_depth, align_speed_with_reference, generate_heart_shape, get_rotation_matrix_xy_plane, get_orient_transform
from hmc.utils.hamer_utils import merge_mano_into_smplx, prepare_hamer, get_hand_ay_verts
try:
    import depth_pro
except:
    depth_pro = None
    print("depth_pro not installed")

CRF = 23  # 17 is lossless, every +6 halves the mp4 size


def parse_args_to_cfg():
    # Put all args to cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="inputs/demo/dance_3.mp4")
    parser.add_argument("--motion_path", type=str, default="")
    parser.add_argument("--reference_path", type=str, default="")
    parser.add_argument("--edit_type", type=str, default='affine_transform', help='edit type: affine_transform')
    parser.add_argument("--affine_transform_args", nargs='+', type=float, default=[0, 0, 0, 0], help='height, width, depth ratios (=0 use gt depth) and rotation angle')
    parser.add_argument("--edit_trajectory_args", nargs='+',  type=int, default=[], help="h w h w 2d trajectory")
    parser.add_argument("--edit_trajectory_as_heart_args", nargs='+', type=float, default=[], help="walk as the heart shape")
    parser.add_argument("--edit_trajectory_kickoff_args", nargs='+',  type=int, default=[], help="h w h w 2d trajectory, might be off the ground")
    parser.add_argument("--repeat_smpl", nargs='+', type=int, default=[0, 0, 0], help='start frmae, end frame (=0 sikp eidt traj) and repeat times (=0 skip align speed)')
    parser.add_argument("--window_size", type=int, default=1, help="window size for moving to the ground")
    parser.add_argument("--output_root", type=str, default=None, help="by default to outputs/demo")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("--track_id", type=int, default=0, help="by default to id 0")
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    parser.add_argument("--from_start_img", action="store_true", help="If true, start trajectory from the original person")
    parser.add_argument("--kid", type=float, default=0, help="If > 0, use smpl/smplx kid. 0 for smpl adult, 1 for kid, support interpolation")
    parser.add_argument("--pause_at_begin", type=int, default=0, help="If > 0, pause at the beignning for x frames")
    parser.add_argument("--pause_at_end", type=int, default=0, help="If > 0, pause at the end for x frames")
    parser.add_argument("--adjust_focal", type=float, default=1.0, help="If != 1, adjust the ay vertices")
    parser.add_argument("--append", type=str, default='', help="Append to the output name")
    parser.add_argument("--displacement", type=float, default=0, help="If > -1000, adjust the displacement with it")
    parser.add_argument("--speed_ratio", type=float, default=1.0, help="adjust the speed with it")
    
    args = parser.parse_args()

    # Input
    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")
    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
        ]

        # Allow to change output root
        if args.output_root is not None:
            overrides.append(f"output_root={args.output_root}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    added_keys = {
        "track_id": args.track_id,
        "motion_path": args.motion_path,
        "reference_path": args.reference_path,
        "edited_output_dir": "",
        "edit_type": args.edit_type,
        "affine_transform_args": args.affine_transform_args,
        "edit_trajectory_args": args.edit_trajectory_args,
        "edit_trajectory_as_heart_args": args.edit_trajectory_as_heart_args,
        "edit_trajectory_kickoff_args": args.edit_trajectory_kickoff_args,
        "repeat_smpl": args.repeat_smpl,
        "window_size": args.window_size,
        "from_start_img": args.from_start_img,
        "kid": args.kid,
        "pause_at_begin": args.pause_at_begin,
        "pause_at_end": args.pause_at_end,
        "adjust_focal": args.adjust_focal,
        "append": args.append,
        "displacement": args.displacement,
        "speed_ratio": args.speed_ratio
    }
    OmegaConf.set_struct(cfg, False)
    for key, value in added_keys.items():
        setattr(cfg, key, value)

    # Output
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Copy raw-input-video to video_path
    Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
    if not Path(cfg.video_path).exists() or get_video_lwh(video_path)[0] != get_video_lwh(cfg.video_path)[0]:
        shutil.copy(video_path, cfg.video_path)

    return cfg


def get_R_T_K_etc(cfg):
    pred = torch.load(cfg.paths.hmr4d_results, weights_only=True)
    smplx = make_smplx("supermotion").cuda()
    
    # get smpl incam
    smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
    pred_c_verts = smplx_out.vertices
    avg_depth = pred_c_verts[0][:, 2].mean()
    
    # get smpl world
    smplx_ay_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = smplx_ay_out.vertices

    K = to_cuda(pred["K_fullimg"][0])
    length, width, height = get_video_lwh(cfg.video_path)

    # initialize depth estimator
    depth_path = os.path.join(cfg.edited_output_dir, 'depth.pt')
    if not os.path.exists(depth_path) and depth_pro is not None:
        depth_model, depth_transform = depth_pro.create_model_and_transforms(device='cuda')
        depth_model = depth_model.eval()
    
    # estimate depth, focallength_calibration, R, T
    # revise GVHMR focal length (i.e., (img_w**2 + img_h**2)**0.5)) with depth. 
    # do not estimate RT after revising pred_c_verts. It is less natural.
    # do not directly revise K, which only changes the ratio of x/z and y/z.
    focallength_calibration = 1
    if cfg.static_cam:
        if depth_pro is not None:
            if not os.path.exists(depth_path):
                depth_image = read_video_np(cfg.video_path, start_frame=0, end_frame=1)[0]
                depth_image = depth_transform(depth_image)
                prediction = depth_model.infer(depth_image, f_px=None)
                torch.save(prediction, depth_path)
            else:
                prediction = torch.load(depth_path, weights_only=True)

            depth_image = prediction["depth"]  # Depth in [m]
            focallength_px = prediction["focallength_px"].item()  # focal length in pixels
            depth_image = torch.clip(depth_image, 1e-4, 100)  # clip depth
            depth_images, focallength_pxs = [depth_image], [focallength_px]
            focallength_calibration = focallength_px / K[0, 0]
            
        R, T = roma.rigid_points_registration(pred_ay_verts[0:1], pred_c_verts[0:1])
    else:
        # use per-frame RT estimation is better if allowed
        depth_images, focallength_pxs = [], []
        if os.path.exists(depth_path) and depth_pro is not None:
            predictions = torch.load(depth_path, weights_only=True)
        else:
            predictions = []
            video = read_video_np(cfg.video_path)

        for idx in range(pred_ay_verts.shape[0]):
            if depth_pro is not None:
                if not os.path.exists(depth_path):
                    depth_image = video[idx]
                    depth_image = depth_transform(depth_image)
                    prediction = depth_model.infer(depth_image, f_px=None)
                    predictions.append(prediction)
                else:
                    prediction = predictions[idx]

                depth_image = prediction["depth"]  # Depth in [m].
                focallength_px = prediction["focallength_px"].item()  # Focal length in pixels.
                depth_image = torch.clip(depth_image, 1e-4, 100)  # 截断无穷远depth
                depth_images.append(depth_image)
                focallength_pxs.append(focallength_px)

        if depth_pro is not None:
            focallength_calibration = (sum(focallength_pxs) / len(focallength_pxs)) / K[0, 0]
            if not os.path.exists(depth_path):
                torch.save(predictions, depth_path)

        R, T = roma.rigid_points_registration(pred_ay_verts, pred_c_verts)

    # print(((pred_ay_verts[0] @ R.T + T) - pred_c_verts[0]).abs().max())

    return R, T, K, width, height, focallength_calibration, avg_depth, depth_images


def get_given_point_global(cfg, R, T, K, width, height, avg_depth, depth_images):
    # get the given point position for affine transformation
    if cfg.affine_transform_args[1] > 0 or cfg.affine_transform_args[1] > 0:
        # edit1: only edit the starting point and keep the original trajectory
        given_point_2d = torch.tensor([int(width * cfg.affine_transform_args[1]), int(height * cfg.affine_transform_args[0])], device="cuda", dtype=torch.float) # move to the left
        if cfg.affine_transform_args[2] > 0:
            depth = avg_depth * cfg.affine_transform_args[2] # 根据人的平均深度来定
        else:
            depth = depth_images[0][int(height * cfg.affine_transform_args[0]), int(width * cfg.affine_transform_args[1])] # 到相机的距离
        given_point_global = incam2world(given_point_2d, R[0], T[0], K, depth)
        given_point_global[1] *= 0
    else:
        given_point_global = 0

    return given_point_global


def get_given_trajectory3d_global(cfg, R, T, K, width, height, focallength_calibration, trajectory_length, avg_depth, depth_images):
    if cfg.edit_type == 'edit_trajectory':
        given_trajectory2d = torch.tensor(cfg.edit_trajectory_args, device="cuda", dtype=torch.float).view(-1, 2).flip(1)
        if cfg.from_start_img:
            origin = torch.tensor([[[0, 0, 0]]], device="cuda", dtype=torch.float)
            origin_2d = detach_to_cpu(get_trajectory2d(origin, cfg, R, T, K, focallength_calibration)).numpy()
            given_trajectory2d = torch.cat((torch.tensor([origin_2d[0, 0]], device="cuda", dtype=torch.float), given_trajectory2d), 0)
        print('2d trajectory key points:', given_trajectory2d)

        # get 3d points
        depth = get_depth(depth_images[0], given_trajectory2d[:, 1], given_trajectory2d[:, 0], avg_depth, cfg)
        given_trajectory3d_global = incam2world(given_trajectory2d, R[0], T[0], K, depth, focallength_calibration)
        given_trajectory3d_global[..., 1] *= 0 # might not be 0 due to inaccurate depth or RT
        # interpolate
        given_trajectory3d_global = interpolate_curve(given_trajectory3d_global, num_points=trajectory_length, s=0, k=2)
    
    elif cfg.edit_type == 'edit_trajectory_as_heart':
        given_trajectory3d_global = generate_heart_shape(h=cfg.edit_trajectory_as_heart_args[0], w=cfg.edit_trajectory_as_heart_args[1], num_points=trajectory_length).to(device="cuda")
    
    elif cfg.edit_type == 'edit_trajectory_kickoff':
        given_trajectory2d =  torch.tensor(cfg.edit_trajectory_kickoff_args, device="cuda", dtype=torch.float).view(-1, 2).flip(1)
        given_trajectory2d = interpolate_curve(given_trajectory2d, num_points=trajectory_length, s=0, k=2)
        depth = torch.arange(trajectory_length, device="cuda") / trajectory_length * avg_depth * 5 + avg_depth
        given_trajectory3d_global = incam2world(given_trajectory2d, R[0], T[0], K, depth, focallength_calibration)
    
    else:
        given_trajectory3d_global = 0
    
    return given_trajectory3d_global


def rotate_and_shift(pred_ay_verts, pred, given_point_global, cfg, R, T):
    rotate_matrix_xy = get_rotation_matrix_xz_plane(torch.tensor(cfg.affine_transform_args[3] * torch.pi), device=pred_ay_verts.device)

    pred_ay_verts = pred_ay_verts @ rotate_matrix_xy.T + given_point_global
    pred_ay_verts, ground_y = move_to_ground(pred_ay_verts, window_size=cfg.window_size) # move to ground

    pred["smpl_params_global"]["global_orient"] = matrix_to_axis_angle(axis_angle_to_matrix(pred["smpl_params_global"]["global_orient"]) @ rotate_matrix_xy)
    pred["smpl_params_global"]["transl"] = pred["smpl_params_global"]["transl"] @ rotate_matrix_xy.T + given_point_global
    pred["smpl_params_global"]["transl"] = move_to_ground_transl(pred["smpl_params_global"]["transl"], ground_y, window_size=cfg.window_size)
    
    pred = transform_smpl_params_incam_from_global(pred, cfg, R, T)

    return pred_ay_verts, pred


def edit_trajectory(pred_ay_verts, pred, given_trajectory3d_global, cfg, R, T):
    # synchronize speed
    transl_global = pred["smpl_params_global"]["transl"].clone()
    if cfg.repeat_smpl[2] != 0:
        given_trajectory3d_global = align_speed_with_reference(transl_global, given_trajectory3d_global)

    # edit orientation
    orient_transform = get_orient_transform(pred["smpl_params_global"]["global_orient"], given_trajectory3d_global)

    # edit trajectory
    pred_ay_verts = pred_ay_verts - transl_global.unsqueeze(1)
    pred_ay_verts = pred_ay_verts @ orient_transform
    pred_ay_verts = pred_ay_verts + transl_global.unsqueeze(1)
    transl_global[..., 1] *= 0
    pred_ay_verts = pred_ay_verts - transl_global.unsqueeze(1) + given_trajectory3d_global.unsqueeze(1)
    pred_ay_verts, ground_y = move_to_ground(pred_ay_verts, window_size=cfg.window_size) # move to ground

    pred["smpl_params_global"]["global_orient"] = matrix_to_axis_angle(axis_angle_to_matrix(pred["smpl_params_global"]["global_orient"]) @ orient_transform.transpose(1, 2))
    pred["smpl_params_global"]["transl"] = pred["smpl_params_global"]["transl"] - transl_global + given_trajectory3d_global
    pred["smpl_params_global"]["transl"] = move_to_ground_transl(pred["smpl_params_global"]["transl"], ground_y, window_size=cfg.window_size)
    pred["smpl_params_global"]["transl"][..., 1] -= 0.75  # bug here, we need to adjust the height

    pred = transform_smpl_params_incam_from_global(pred, cfg, R, T)

    return pred_ay_verts, pred


def edit_trajectory_kickoff(pred_ay_verts, pred, given_trajectory3d_global, cfg, R, T):
    transl_global = pred["smpl_params_global"]["transl"].clone()
    trajectory_length = transl_global.shape[0]

    # edit orientation
    angles = torch.arange(trajectory_length, device="cuda") / trajectory_length * 3.14 * 2
    orient_transform = torch.stack([get_rotation_matrix_xy_plane(angle, device=pred_ay_verts.device) for angle in angles])

    # edit trajectory
    pred_ay_verts = pred_ay_verts - transl_global.unsqueeze(1)
    pred_ay_verts = pred_ay_verts @ orient_transform
    pred_ay_verts = pred_ay_verts + transl_global.unsqueeze(1)
    given_trajectory3d_global = given_trajectory3d_global - given_trajectory3d_global[0] # start from origin
    pred_ay_verts = pred_ay_verts + given_trajectory3d_global.unsqueeze(1)

    # pasuse at the beginning
    pause_at_begin = 25
    pred_ay_verts = torch.cat([pred_ay_verts[0:1]] * pause_at_begin + [pred_ay_verts], 0)

    pred["smpl_params_global"]["global_orient"] = matrix_to_axis_angle(axis_angle_to_matrix(pred["smpl_params_global"]["global_orient"]) @ orient_transform.transpose(1, 2))
    pred["smpl_params_global"]["transl"] = pred["smpl_params_global"]["transl"] - transl_global + given_trajectory3d_global

    pred = transform_smpl_params_incam_from_global(pred, cfg, R, T)

    return pred_ay_verts, pred


def edit(cfg, render_hamer=False):
    if os.path.exists(os.path.join(cfg.motion_path, 'sample_rep00.mp4')):
        fps = get_video_fps(os.path.join(cfg.motion_path, 'sample_rep00.mp4'))  # from text-to-motion model FlowMDM
    elif os.path.exists(os.path.join(cfg.motion_path, '0_input_video.mp4')):
        fps = get_video_fps(os.path.join(cfg.motion_path, '0_input_video.mp4')) # from videos estimated by GVHMR
    else:
        fps = 25.0

    # for action
    pred = torch.load(os.path.join(cfg.motion_path, 'hmr4d_results.pt'), weights_only=True)

    # for background
    R, T, K, width, height, focallength_calibration, avg_depth, depth_images = get_R_T_K_etc(cfg)
    
    # for action
    if cfg.kid > 0:
        # adjust the speed. Suppose baby is speed_ratio of the speed of adult
        cfg.speed_ratio = 1/4
        pred['smpl_params_global']['transl'] = (pred['smpl_params_global']['transl'] - pred['smpl_params_global']['transl'][0:1]) * (1 + cfg.kid * (cfg.speed_ratio - 1)) + pred['smpl_params_global']['transl'][0:1]
    else:
        pred['smpl_params_global']['transl'] = (pred['smpl_params_global']['transl'] - pred['smpl_params_global']['transl'][0:1]) * cfg.speed_ratio + pred['smpl_params_global']['transl'][0:1]
    flat_hand_mean = False
    use_pca = True
    hand_info = None
    if render_hamer and os.path.exists(f"{cfg.motion_path}/hamer_info.pkl"):
        hand_info = pickle.load(open(f"{cfg.motion_path}/hamer_info.pkl", "rb"))
        if 'all_pred_mano_params' in hand_info[0]:
            pred["smpl_params_global"] = merge_mano_into_smplx(pred["smpl_params_global"], hand_info, (R.repeat(pred["smpl_params_global"]['global_orient'].shape[0], 1, 1) if cfg.static_cam else R).cpu().numpy()) # use projection for hand_info as it is incam
            flat_hand_mean = True
            use_pca = False
            hand_info = None
            print('merge mano to smplx')
    pred, hand_info = clip_and_repeat_smpl(pred, cfg, hand_info)
    pred, hand_info = repeat_smpl_at_begin_end(pred, cfg, hand_info)
    trajectory_length = pred["smpl_params_global"]["body_pose"].shape[0]
    
    # for reference
    if cfg.reference_path != "":
        if pred["smpl_params_global"]["body_pose"].shape[1] == 63: # smplx
            # if ref smpl is estimated from GVHMR
            reference_pred = torch.load(os.path.join(cfg.reference_path, 'hmr4d_results.pt'), weights_only=True)
            pred["smpl_params_global"]["betas"] = reference_pred["smpl_params_global"]["betas"][0:1, :].repeat(pred["smpl_params_global"]["betas"].shape[0], 1)
        elif pred["smpl_params_global"]["body_pose"].shape[1] == 69: # smpl
            pred["smpl_params_incam"] = copy.deepcopy(pred["smpl_params_global"])
            # if ref smpl is estimated from 4D-Humans
            reference_pred = np.load(os.path.join(cfg.reference_path, "0_input_video.npy"), allow_pickle=True).item()
            pred["smpl_params_global"]["betas"] = torch.from_numpy(reference_pred["smpls"]["betas"]).repeat(pred["smpl_params_global"]["betas"].shape[0], 1) 

    if cfg.kid > 0:
        pred["smpl_params_incam"]["betas"] = pred["smpl_params_global"]["betas"] = torch.cat((pred["smpl_params_global"]["betas"], pred["smpl_params_global"]["betas"][:, :1] * 0 + cfg.kid), 1)

    # get smpl model
    if pred["smpl_params_global"]["body_pose"].shape[1] == 63: # smplx
        kwargs = {'age': 'kid','kid_template_path': 'hmc/body_model/smplx_kid_template.npy'} if cfg.kid > 0 else dict() # ref: https://github.com/pixelite1201/agora_evaluation/blob/master/docs/kid_model.md
        kwargs.update({'use_pca': use_pca, 'flat_hand_mean': flat_hand_mean})
        smpl_model = make_smplx("supermotion", **kwargs).cuda()
    elif pred["smpl_params_global"]["body_pose"].shape[1] == 69:
        kwargs = {'age': 'kid','kid_template_path': 'hmc/body_model/smpl_kid_template.npy'} if cfg.kid > 0 else dict()
        smpl_model = make_smplx("smpl", **kwargs).cuda() # smpl
    
    # get human vertices
    pred["smpl_params_global"] = to_cuda(pred["smpl_params_global"])
    smplx_ay_out = smpl_model(**pred["smpl_params_global"])
    pred_ay_verts = smplx_ay_out.vertices

    # get R, T for hands
    if render_hamer:
        if hand_info is not None:
            # get RT for hand from the motion video
            if cfg.static_cam:
                R_hand, T_hand = roma.rigid_points_registration(pred_ay_verts[0:1], smpl_model(**to_cuda(pred["smpl_params_incam"])).vertices[0:1])
            else:
                R_hand, T_hand = roma.rigid_points_registration(pred_ay_verts, smpl_model(**to_cuda(pred["smpl_params_incam"])).vertices)
        else:
            R_hand, T_hand = None, None


    ######## Case 1: affine transformation ############# 
    # rotate and shift all vertices with one angle (cfg.edit[3]) and one shift (given_point_global) (equals to moving camera R T)
    if cfg.edit_type == 'affine_transform':
        given_point_global = get_given_point_global(cfg, R, T, K, width, height, avg_depth, depth_images)
        pred_ay_verts, pred = rotate_and_shift(pred_ay_verts, pred, given_point_global, cfg, R, T)
    
    ######## Case 2: edit trajectory (on the ground) ############# 
    ######## Case 3: edit trajectory as heart shape ############# 
    # edit vertices, not smpl paras
    if cfg.edit_type in ['edit_trajectory', 'edit_trajectory_as_heart']:
        given_trajectory3d_global = get_given_trajectory3d_global(cfg, R, T, K, width, height, focallength_calibration, trajectory_length, avg_depth, depth_images)
        pred_ay_verts, pred = edit_trajectory(pred_ay_verts, pred, given_trajectory3d_global, cfg, R, T)

    ######## Case 4: edit trajectory (off the ground) ############# 
    if cfg.edit_type == 'edit_trajectory_kickoff':
        given_trajectory3d_global = get_given_trajectory3d_global(cfg, R, T, K, width, height, focallength_calibration, trajectory_length, avg_depth, depth_images)
        pred_ay_verts, pred = edit_trajectory_kickoff(pred_ay_verts, pred, given_trajectory3d_global, cfg, R, T)
    

    # project to the incam space
    pred_c_verts_edited = world2incam(pred_ay_verts, R, T, cfg, focallength_calibration)

    # matching hands
    if render_hamer:
        hand_ay_verts = get_hand_ay_verts(pred_ay_verts, R, T, K, cfg, focallength_calibration, hand_info, R_hand, T_hand)   
        hand_c_verts_edited = world2incam(hand_ay_verts, R, T, cfg, focallength_calibration) # w2c before creating the camera for focal calibration
    else:
        hand_c_verts_edited = None

    return pred_c_verts_edited, hand_c_verts_edited, smpl_model, pred, width, height, fps, R, T, K, focallength_calibration


def render_incam_edited(cfg, pred_c_verts_edited, hand_c_verts_edited, smpl_model, pred, width, height, fps, R, T, K, focallength_calibration):
    reader = get_video_reader(cfg.video_path)  # (F, H, W, 3), uint8, numpy
    renderer = Renderer(width, height, device="cuda", faces=smpl_model.faces, K=K)

    trajectory2d = detach_to_cpu(get_trajectory2d(pred["smpl_params_global"]["transl"], cfg, R, T, K, focallength_calibration)).numpy().astype(np.int32)
    bbx_xyxy_render = get_bbox_from_vertices(pred_c_verts_edited, K) # apply the same transformation to bbox

    # -- render mesh -- #
    incam_video_path = Path(f'{cfg.edited_output_dir}/1_incam.mp4')
    writer = get_writer(incam_video_path, fps=fps, crf=CRF)
    
    if cfg.static_cam:
        img_raw = read_video_np(cfg.video_path, start_frame=0, end_frame=1)[0]
        for i in tqdm(list(range(pred_c_verts_edited.shape[0])), total=pred_c_verts_edited.shape[0], desc=f"Rendering Incam"):
            img = img_raw[:,:]
            color = [0.9, 0.9, 0.9]
            # Draw the trajectory by connecting the points
            for j in range(len(trajectory2d) - 1):
                cv2.line(img, tuple(trajectory2d[j]), tuple(trajectory2d[j + 1]), (255, 0, 0), 8)
            for j in range(len(trajectory2d) - 1):
                if j < i:
                    cv2.line(img, tuple(trajectory2d[j]), tuple(trajectory2d[j + 1]), (0, 255, 0), 2)
            try:
                img = renderer.render_mesh(pred_c_verts_edited[i], img, color) # from incam verts

                # bbx
                bbx_xyxy_ = bbx_xyxy_render[i].cpu().numpy()
                lu_point = bbx_xyxy_[:2].astype(int)
                rd_point = bbx_xyxy_[2:].astype(int)
                img = cv2.rectangle(img, lu_point, rd_point, (255, 178, 102), 2)
            except:
                pass
            writer.write_frame(img)

    else:
        background_video = read_video_np(cfg.video_path)
        for i in tqdm(list(range(pred_c_verts_edited.shape[0])), total=pred_c_verts_edited.shape[0], desc=f"Rendering Incam"):
            
            img = background_video[i]
            color = [0.9, 0.9, 0.9]
            # Draw the trajectory by connecting the points
            for j in range(len(trajectory2d) - 1):
                cv2.line(img, tuple(trajectory2d[j]), tuple(trajectory2d[j + 1]), (255, 0, 0), 8)
            for j in range(len(trajectory2d) - 1):
                if j < i:
                    cv2.line(img, tuple(trajectory2d[j]), tuple(trajectory2d[j + 1]), (0, 255, 0), 2)
            try:
                img = renderer.render_mesh(pred_c_verts_edited[i], img, color) # from incam verts

                # bbx
                bbx_xyxy_ = bbx_xyxy_render[i].cpu().numpy()
                lu_point = bbx_xyxy_[:2].astype(int)
                rd_point = bbx_xyxy_[2:].astype(int)
                img = cv2.rectangle(img, lu_point, rd_point, (255, 178, 102), 2)
            except:
                pass
            writer.write_frame(img)
        
    writer.close()
    reader.close()

    # save smpl para
    pred["smpl_params_global"] = detach_to_cpu(pred["smpl_params_global"])
    pred["smpl_params_incam"] = detach_to_cpu(pred["smpl_params_incam"])
    torch.save(pred, f'{cfg.edited_output_dir}/pred.pt')
    os.makedirs(f'{cfg.edited_output_dir}/preprocess', exist_ok=True)
    torch.save(detach_to_cpu({'bbx_xyxy': bbx_xyxy_render}), f'{cfg.edited_output_dir}/preprocess/bbx.pt')


def render_depth_normal_cs_map_edited(cfg, pred_c_verts_edited, hand_c_verts_edited, smpl_model, pred, width, height, fps):
    if hand_c_verts_edited is not None:
        hand_colors, hand_faces = prepare_hamer()
        renderer = ConditionRenderer(width, height, device="cuda", faces=smpl_model.faces, K=K, hand_faces=hand_faces)
    else:
        renderer = ConditionRenderer(width, height, device="cuda", faces=smpl_model.faces, K=K)

    color_map_path = os.path.join(cfg.edited_output_dir, "cs_map.mp4")
    color_map_writer = get_writer(color_map_path, fps=fps, crf=CRF)

    normal_map_path = os.path.join(cfg.edited_output_dir, "normal_map.mp4")
    normal_map_writer = get_writer(normal_map_path, fps=fps, crf=CRF)

    depth_path = os.path.join(cfg.edited_output_dir, "depth_map.mp4")
    depth_writer = get_writer(depth_path, fps=fps, crf=CRF)

    hamer_path = os.path.join(cfg.edited_output_dir, "hamer.mp4")
    hamer_writer = get_writer(hamer_path, fps=fps, crf=CRF)

    if pred["smpl_params_global"]["body_pose"].shape[1] == 63: # smplx
        smplx_color = torch.load("hmc/body_model/smplx_color.pt", weights_only=False)
        smplx_cs_colors = torch.from_numpy(smplx_color) / 255.0  # V,3, jingkai's requirement
    elif pred["smpl_params_global"]["body_pose"].shape[1] == 69:
        smplx_color = torch.load("hmc/body_model/smpl_color.pt", weights_only=False)
        smplx_cs_colors = smplx_color

    for i in tqdm(list(range(pred_c_verts_edited.shape[0])), total=pred_c_verts_edited.shape[0], desc=f"Rendering Condition Maps"):
        try:
            if hand_c_verts_edited is not None:
                # img_raw = read_video_np(cfg.video_path, start_frame=i, end_frame=i+1)[0]
                img, normal, depth, hamer = renderer.render_mesh(
                    pred_c_verts_edited[i], None, smplx_cs_colors, hand_vertices=hand_c_verts_edited[i], hand_colors=hand_colors,
                )
            else:
                img, normal, depth, hamer = renderer.render_mesh(
                    pred_c_verts_edited[i], None, smplx_cs_colors,
                )
        except:
            img = normal = depth = hamer = np.zeros((renderer.height, renderer.width, 3), dtype=np.uint8)
            
        color_map_writer.write_frame(img)
        normal_map_writer.write_frame(normal)
        depth_writer.write_frame(depth)
        hamer_writer.write_frame(hamer)

    color_map_writer.close()
    normal_map_writer.close()
    depth_writer.close()
    hamer_writer.close()


if __name__ == "__main__":
    cfg = parse_args_to_cfg()
    paths = cfg.paths
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f'[GPU]: {torch.cuda.get_device_properties("cuda")}')

    # ===== Preprocess and save to disk ===== #
    run_preprocess(cfg)
    data = load_data_dict(cfg)

    # ===== HMR4D ===== #
    if not Path(paths.hmr4d_results).exists():
        Log.info("[HMR4D] Predicting")
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        tic = Log.sync_time()
        pred = model.predict(data, static_cam=cfg.static_cam)
        pred = detach_to_cpu(pred)
        data_time = data["length"] / 30
        Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s")
        torch.save(pred, paths.hmr4d_results)

    output_dir = cfg.output_dir + f'_{os.path.basename(cfg.motion_path)}'
    if cfg.reference_path != "":
        output_dir = output_dir + f'_{os.path.basename(cfg.reference_path)}'
    if cfg.append != '':
        output_dir = output_dir + f'_{cfg.append}'
    os.makedirs(output_dir, exist_ok=True)
    cfg.edited_output_dir = output_dir

    # ===== Edit ===== #
    # does not support motion from FlowMDM when render_hamer=True
    pred_c_verts_edited, hand_c_verts_edited, smpl_model, pred, width, height, fps, R, T, K, focallength_calibration = edit(cfg, render_hamer=True)

    # ===== Render ===== #
    render_incam_edited(cfg, pred_c_verts_edited, hand_c_verts_edited, smpl_model, pred, width, height, fps, R, T, K, focallength_calibration)    
    render_depth_normal_cs_map_edited(cfg, pred_c_verts_edited, hand_c_verts_edited, smpl_model, pred, width, height, fps) 
    
    