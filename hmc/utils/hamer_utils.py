# Reference: https://github.com/VincentHu19/Mano2Smpl-X

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import roma
import pickle


def compute_global_rotation(pose_axis_anges, joint_idx):
    """
    Calculating joints' global rotation.

    Args:
        pose_axis_anges (np.array): SMPLX's local pose (22, 3)
        joint_idx (int): Joint index to calculate the global rotation for.

    Returns:
        np.array: (3, 3) Global rotation matrix.
    """
    global_rotation = np.eye(3)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    while joint_idx != -1:
        joint_rotation = R.from_rotvec(pose_axis_anges[joint_idx]).as_matrix()
        global_rotation = joint_rotation @ global_rotation
        joint_idx = parents[joint_idx]
    return global_rotation


def merge_mano_into_smplx(gvhmr_smplx_params, all_pred_mano_params, RR=None):
    """
    Merge MANO parameters into SMPL-X parameters.

    Args:
        gvhmr_smplx_params (dict): Dictionary containing SMPL-X parameters.
        all_pred_mano_params (list): List containing MANO parameters for each sample.
        RR (np.array, optional): Optional rotation matrix for mirroring.

    Returns:
        dict: Updated SMPLX parameters with merged MANO configurations.
    """
    M = np.diag([-1, 1, 1])  # Transformation matrix for flipping.

    # Preparing containers for left and right hand poses.
    gvhmr_smplx_params["left_hand_pose"] = []  
    gvhmr_smplx_params["right_hand_pose"] = []

    for index in range(gvhmr_smplx_params["global_orient"].shape[0]):
        # Initialize left and right hand pose vectors
        left_hand_pose = np.zeros(45)
        right_hand_pose = np.zeros(45)

        try:
            hamer_mano_params = all_pred_mano_params[index]['all_pred_mano_params']
            full_body_pose = torch.concatenate(
                (
                    gvhmr_smplx_params["global_orient"][index].unsqueeze(0),
                    gvhmr_smplx_params["body_pose"][index].reshape(21, 3)
                ),
                dim=0
            )

            # Compute global rotations for left and right elbow
            left_elbow_global_rot = compute_global_rotation(full_body_pose, 18)  # Left elbow idx: 18
            right_elbow_global_rot = compute_global_rotation(full_body_pose, 19)  # Right elbow idx: 19

            # Determine the hand used for orientation (right or left first)
            right_first = (all_pred_mano_params[index]['all_right'][0].item() == 1)

            # Left wrist global rotation and flipping
            left_wrist_global_rot = hamer_mano_params[1 if right_first else 0]["global_orient"][0]
            left_wrist_global_rot = M @ left_wrist_global_rot @ M  # Represent left hand as right hand by flipping

            # Apply mirroring rotation matrix if provided
            if RR is not None:
                left_wrist_global_rot = RR[index] @ left_wrist_global_rot

            # Compute left wrist pose relative to the elbow
            left_wrist_pose = np.linalg.inv(left_elbow_global_rot) @ left_wrist_global_rot

            # Right wrist global rotation
            right_wrist_global_rot = hamer_mano_params[0 if right_first else 1]["global_orient"][0]
            if RR is not None:
                right_wrist_global_rot = RR[index] @ right_wrist_global_rot

            # Compute right wrist pose relative to the elbow
            right_wrist_pose = np.linalg.inv(right_elbow_global_rot) @ right_wrist_global_rot

            # Convert wrist poses to rotation vectors
            left_wrist_pose_vec = R.from_matrix(left_wrist_pose).as_rotvec()
            right_wrist_pose_vec = R.from_matrix(right_wrist_pose).as_rotvec()

            # Process finger poses for both hands
            for i in range(15):
                left_finger_pose = M @ hamer_mano_params[1 if right_first else 0]["hand_pose"][i] @ M
                left_finger_pose_vec = R.from_matrix(left_finger_pose).as_rotvec()
                left_hand_pose[i * 3: i * 3 + 3] = left_finger_pose_vec

                right_finger_pose = hamer_mano_params[0 if right_first else 1]["hand_pose"][i]
                right_finger_pose_vec = R.from_matrix(right_finger_pose).as_rotvec()
                right_hand_pose[i * 3: i * 3 + 3] = right_finger_pose_vec

            # Update SMPL-X body poses with wrist information
            gvhmr_smplx_params["body_pose"][index, 57:60] = torch.from_numpy(left_wrist_pose_vec).float()
            gvhmr_smplx_params["body_pose"][index, 60:63] = torch.from_numpy(right_wrist_pose_vec).float()

        except Exception as e:
            # Handle exceptions (e.g., missing parameters or undefined behavior)
            print(f"Error processing index {index}: {e}")
            pass

        # Append the processed hand poses
        gvhmr_smplx_params["left_hand_pose"].append(torch.from_numpy(left_hand_pose).float())
        gvhmr_smplx_params["right_hand_pose"].append(torch.from_numpy(right_hand_pose).float())

    # Stack the hand poses to form tensors
    gvhmr_smplx_params["left_hand_pose"] = torch.stack(gvhmr_smplx_params["left_hand_pose"])
    gvhmr_smplx_params["right_hand_pose"] = torch.stack(gvhmr_smplx_params["right_hand_pose"])

    return gvhmr_smplx_params


def prepare_hamer():
    idxs_data = pickle.load(open('hmc/body_model/MANO_SMPLX_vertex_ids.pkl', 'rb'))
    hand_idxs = np.concatenate([idxs_data['left_hand'], idxs_data['right_hand']])
    left_hand_colors = np.tile(np.array([0.5, 1.0, 0.5]).reshape((1, 1, 3)), (1, idxs_data['left_hand'].shape[0], 1))
    right_hand_colros = np.tile(np.array([1.0, 0.5, 0.5]).reshape((1, 1, 3)), (1, idxs_data['right_hand'].shape[0], 1))
    hand_colors = torch.from_numpy(np.concatenate([left_hand_colors, right_hand_colros], axis=1)).to(dtype=torch.float32)
    
    # creat Hash Mapping for fast search
    hand_set = set(hand_idxs.tolist())
    old_to_new = {old: new for new, old in enumerate(hand_idxs)}

    # filter valid faces and transform the indexes
    faces = torch.from_numpy(np.load('hmc/body_model/smplx_faces.npy')).unsqueeze(0)
    hand_faces = []
    for face in faces[0]:
        v0, v1, v2 = face
        v0, v1, v2 = v0.item(), v1.item(), v2.item()
        # check if all vertices are in the target index set
        if all(v in hand_set for v in [v0, v1, v2]):
            # move to the new index system
            new_face = [old_to_new[v0], old_to_new[v1], old_to_new[v2]]
            hand_faces.append(new_face)

    hand_faces = np.array(hand_faces, dtype=np.int64)[None]  # [1, Y, 3]
    
    # suture hand faces
    faces_new = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279], [122, 118, 279],
                            [279, 118, 215], [118, 117, 215], [215, 117, 214], [117, 119, 214], [214, 119, 121],
                            [119, 120, 121], [121, 120, 78], [120, 108, 78], [78, 108, 79]]).reshape((1, -1, 3))
    hand_faces = torch.from_numpy(np.concatenate([hand_faces, faces_new], axis=1)).to(dtype=torch.long)

    return hand_colors.squeeze(0), hand_faces.squeeze(0)


def get_hand_ay_verts(vertices, R, T, K, cfg, focallength_calibration, hand_info=None, R_hand=None, T_hand=None):
    nframe = vertices.shape[0]

    if hand_info is not None:
        # get hand vertices
        idxs_data = pickle.load(open('hmc/body_model/MANO_SMPLX_vertex_ids.pkl', 'rb'))
        hand_idxs = np.concatenate([idxs_data['left_hand'], idxs_data['right_hand']])
        left_hand_idxs = idxs_data['left_hand']
        right_hand_idxs = idxs_data['right_hand']

        left_hand_vertices = []
        right_hand_vertices = []
        left_hand_camts = []
        right_hand_camts = []
        valid_hand = []
        for iframe in np.arange(nframe):
            if hand_info[iframe] is None or len(hand_info[iframe]['all_right']) != 2:  # fail without two hands
                left_hand_vertices.append(np.zeros((778, 3), dtype=np.float32))
                right_hand_vertices.append(np.zeros((778, 3), dtype=np.float32))
                valid_hand.append(False)
            elif hand_info[iframe]['all_right'][0] + hand_info[iframe]['all_right'][1] != 1:  # fail without just one left hand and one right hand
                left_hand_vertices.append(np.zeros((778, 3), dtype=np.float32))
                right_hand_vertices.append(np.zeros((778, 3), dtype=np.float32))
                valid_hand.append(False)
            else:
                for j in range(len(hand_info[iframe]['all_right'])):
                    is_right = hand_info[iframe]['all_right'][j].item()
                    if is_right == 1:
                        right_hand_vertices.append(hand_info[iframe]['all_verts'][j])
                        right_hand_camts.append(hand_info[iframe]['all_cam_t'][j])
                    else:
                        left_hand_vertices.append(hand_info[iframe]['all_verts'][j])
                        left_hand_camts.append(hand_info[iframe]['all_cam_t'][j])
                valid_hand.append(True)

        print(f"Using hamer, valid hands: {np.sum(valid_hand)}/{nframe}")

        smpl_left_hand_vertices = vertices[:, left_hand_idxs].to(dtype=torch.float32)
        smpl_right_hand_vertices = vertices[:, right_hand_idxs].to(dtype=torch.float32)

        left_hand_vertices = torch.tensor(np.array(left_hand_vertices), dtype=torch.float32, device=vertices.device)
        right_hand_vertices = torch.tensor(np.array(right_hand_vertices), dtype=torch.float32, device=vertices.device)
        left_hand_camts = torch.tensor(np.array(left_hand_camts), dtype=torch.float32, device=vertices.device)
        right_hand_camts = torch.tensor(np.array(right_hand_camts), dtype=torch.float32, device=vertices.device)

        # use hand vertices for matching
        for j in range(len(valid_hand)):
            if valid_hand[j]:
                R1, T1, s1 = roma.rigid_points_registration(left_hand_vertices[j],
                                                            smpl_left_hand_vertices[j],
                                                            compute_scaling=True)  # weights=None
                left_hand_vertices[j] = (R1 @ (left_hand_vertices[j] * s1).mT).mT + T1[None]
                R2, T2, s2 = roma.rigid_points_registration(right_hand_vertices[j],
                                                            smpl_right_hand_vertices[j],
                                                            compute_scaling=True)  # weights=None
                right_hand_vertices[j] = (R2 @ (right_hand_vertices[j] * s2).mT).mT + T2[None]
        
        valid_hand = torch.tensor(valid_hand).to(dtype=torch.float32, device=vertices.device)[:, None, None]  # [F,1,1]
        left_hand_vertices = left_hand_vertices * valid_hand + smpl_left_hand_vertices * (1 - valid_hand)
        right_hand_vertices = right_hand_vertices * valid_hand + smpl_right_hand_vertices * (1 - valid_hand)
        hand_vertices = torch.cat([left_hand_vertices, right_hand_vertices], dim=1)
    else:  
        # directly copy smpl hands
        print('Using smplx hands')
        idxs_data = pickle.load(open('hmc/body_model/MANO_SMPLX_vertex_ids.pkl', 'rb'))
        hand_idxs = np.concatenate([idxs_data['left_hand'], idxs_data['right_hand']])
        hand_vertices = vertices[:, hand_idxs].to(dtype=torch.float32)
    
    return hand_vertices
