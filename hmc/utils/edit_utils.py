import torch
import numpy as np
from scipy.interpolate import interp1d
from pytorch3d.transforms import quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle, axis_angle_to_quaternion, quaternion_to_axis_angle


def clip_and_repeat_smpl(pred, cfg, hand_info=None):
    if cfg.repeat_smpl[2] != 0:
        
        start, end, repeat = cfg.repeat_smpl[0], cfg.repeat_smpl[1], cfg.repeat_smpl[2]
        end = len(pred['smpl_params_global']['transl']) if end == -1 else end + 1 

        for key in ['transl', 'body_pose', 'global_orient', 'betas']:
            pred['smpl_params_global'][key] = torch.cat([pred['smpl_params_global'][key][start: end]] * repeat, 0)
            pred['smpl_params_incam'][key] = torch.cat([pred['smpl_params_incam'][key][start: end]] * repeat, 0)

        if 'left_hand_pose' in pred['smpl_params_global']:
            for key in ['left_hand_pose', 'right_hand_pose']:
                pred['smpl_params_global'][key] = torch.cat([pred['smpl_params_global'][key][start: end]] * repeat, 0)

        frames = end - start
        for i in [0, 2]:
            pred['smpl_params_global']['transl'][:, i] = pred['smpl_params_global']['transl'][:, i] - pred['smpl_params_global']['transl'][0:1, i]
            for j in range(1, repeat):
                displacement = pred['smpl_params_global']['transl'][j*frames-1, i] + cfg.displacement # do not use (2*(j-1) - (j-2)) to avoid slipping
                pred['smpl_params_global']['transl'][j*frames: (j+1)*frames, i] = pred['smpl_params_global']['transl'][j*frames: (j+1)*frames, i] + displacement
            pred['smpl_params_global']['transl'][:, i] = moving_average(pred['smpl_params_global']['transl'][:, i], window_size=1) # avoid feet slipping
        # only smooth ['transl'][:, 1] with a small window as it leads to unnatural action. The sudden change of smpl is not a problem for video generation.
        pred['smpl_params_global']['transl'][:, 1] = moving_average(pred['smpl_params_global']['transl'][:, 1], window_size=1)

        if hand_info is not None:
            hand_info = np.concatenate([hand_info[start:end]] * repeat, 0)

    return pred, hand_info


def repeat_smpl_at_begin_end(pred, cfg, hand_info=None):
    if cfg.pause_at_begin > 0:
        for key in ['transl', 'body_pose', 'global_orient', 'betas']:
            pred['smpl_params_global'][key] = torch.cat([pred['smpl_params_global'][key][:1]] * cfg.pause_at_begin + [pred['smpl_params_global'][key]], 0)
            pred['smpl_params_incam'][key] = torch.cat([pred['smpl_params_incam'][key][:1]] * cfg.pause_at_begin + [pred['smpl_params_incam'][key]], 0)
        
        if 'left_hand_pose' in pred['smpl_params_global']:
            for key in ['left_hand_pose', 'right_hand_pose']:
                pred['smpl_params_global'][key] = torch.cat([pred['smpl_params_global'][key][:1]] * cfg.pause_at_begin + [pred['smpl_params_global'][key]], 0)

        if hand_info is not None:
            hand_info = np.concatenate([hand_info[:1]] * cfg.pause_at_begin + [hand_info], 0)

    if cfg.pause_at_end > 0:
        for key in ['transl', 'body_pose', 'global_orient', 'betas']:
            pred['smpl_params_global'][key] = torch.cat([pred['smpl_params_global'][key]] + [pred['smpl_params_global'][key][-1:]] * cfg.pause_at_end, 0)
            pred['smpl_params_incam'][key] = torch.cat([pred['smpl_params_incam'][key]] + [pred['smpl_params_incam'][key][-1:]] * cfg.pause_at_end, 0)
        
        if 'left_hand_pose' in pred['smpl_params_global']:
            for key in ['left_hand_pose', 'right_hand_pose']:
                pred['smpl_params_global'][key] = torch.cat([pred['smpl_params_global'][key]] + [pred['smpl_params_global'][key][-1:]] * cfg.pause_at_end, 0)

        if hand_info is not None:
            hand_info = np.concatenate([hand_info] + [hand_info[-1:]] * cfg.pause_at_end, 0)
    
    return pred, hand_info


def points_padding(points):
    padding = torch.ones_like(points)[..., 0:1]
    points = torch.cat([points, padding], dim=-1)
    return points


def interpolate_curve(control_points, num_points=100, s=0, k=2):
    """
    Interpolates points along a curve using B-splines.
    Args:
    - control_points (np.array): An array of shape (N, 2) representing the control points.
    - num_points (int): Number of points to sample along the curve.
    
    Returns:
    - sampled_points (np.array): An array of shape (num_points, 2) of evenly spaced points on the curve.
    """
    from scipy.interpolate import splprep, splev

    points = np.array(control_points.cpu().numpy())
    tck, u = splprep([points[:, i] for i in range(points.shape[1])], s=s, k=k)
    u_new = np.linspace(0, 1, num_points)
    new_points = splev(u_new, tck)
    
    return torch.from_numpy(np.vstack(new_points).T).to(device=control_points.device, dtype=control_points.dtype)


def generate_heart_shape(h=3, w=2, num_points=1000):
    # Generate values of t
    t = torch.linspace(0, 2 * torch.pi, num_points)
    
    # Parametric equations for the heart shape
    x = 16 * torch.sin(t)**3
    y = 13 * torch.cos(t) - 5 * torch.cos(2*t) - 2 * torch.cos(3*t) - torch.cos(4*t)

    x = x / 16 * h
    y = y / 16 * w

    y -= y.min()
    
    return torch.stack((x, torch.zeros_like(x), y), -1)


def move_to_start_point_face_z(verts, J_regressor):
        "XZ to origin, Start from the ground, Face-Z"
        # position
        verts = verts.clone()  # (L, V, 3)
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
        offset[1] = verts[:, :, [1]].min()
        verts = verts - offset
        # face direction
        T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts


def move_to_ground(verts, window_size=0):
    if window_size > 0:
        b, n, c = verts.shape
        group = b // window_size + 1
        to_pad = group * window_size - b
        to_pad_end = ((group - 1) * window_size)
        verts = torch.concat((verts,  verts[to_pad_end - to_pad: to_pad_end]))
        verts = verts.view(group, window_size * n, c)

        ground_y = verts[:, :, 1].min(dim=1, keepdim=True)[0] # find the smallest y of all points as the ground y (the feet y cannot be smaller than ground y)
        verts[..., 1] -= ground_y # In FlowMDM, we have centerred it to origin point. We now move foot to the ground
        verts = verts.view(group * window_size, n, c)[:b]
    else:
        ground_y = None

    return verts, ground_y


def move_to_ground_transl(transl, ground_y, window_size=0):
    if window_size > 0:
        b, c = transl.shape
        group = b // window_size + 1
        to_pad = group * window_size - b
        to_pad_end = ((group - 1) * window_size)

        transl = torch.concat((transl,  transl[to_pad_end - to_pad: to_pad_end]))
        transl = transl.view(group, window_size, c)
        transl[..., 1] -= ground_y[:group] # In FlowMDM, we have centerred it to origin point. We now move foot to the ground
        transl = transl.view(group * window_size, c)[:b]

    return transl


def move_to_ground_mesh(hand_verts, ground_y, window_size=0):
    if window_size > 0:
        b, n, c = hand_verts.shape
        group = b // window_size + 1
        to_pad = group * window_size - b
        to_pad_end = ((group - 1) * window_size)

        hand_verts = torch.concat((hand_verts,  hand_verts[to_pad_end - to_pad: to_pad_end]))
        hand_verts = hand_verts.view(group, window_size * n, c)
        hand_verts[..., 1] -= ground_y[:group] # In FlowMDM, we have centerred it to origin point. We now move foot to the ground
        hand_verts = hand_verts.view(group * window_size, n, c)[:b]

    return hand_verts


def get_rotation_matrix_xz_plane(theta, device='cpu'):
    return torch.tensor([
        [torch.cos(theta), 0, -torch.sin(theta)],
        [0,              1,             0],
        [torch.sin(theta),  0, torch.cos(theta)],
    ], device=device)


def get_rotation_matrix_xy_plane(theta, device='cpu'):
    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta), torch.cos(theta),  0],
        [0,  0, 1],
    ], device=device)


def get_bbox_from_vertices(verts, K):
    verts = torch.div(verts, verts[..., 2:])
    verts2d = torch.matmul(K, verts.transpose(-1, -2)).transpose(-1, -2)[..., :2] # nx(wh)

    min_h = verts2d[..., 0].min(dim=1)[0]
    max_h = verts2d[..., 0].max(dim=1)[0]
    min_w = verts2d[..., 1].min(dim=1)[0]
    max_w = verts2d[..., 1].max(dim=1)[0]

    return torch.stack((min_h, min_w, max_h, max_w), dim=-1)


def get_trajectory2d(trajectory3d, cfg, R, T, K, focallength_calibration):
    trajectory3d = trajectory3d.clone()
    if cfg.edit_type != 'edit_trajectory_kickoff':
        trajectory3d[..., 1] *= 0
    if cfg.static_cam:
        trajectory3d = trajectory3d @ R[0].T + T[0]
    else:
        for i in range(trajectory3d.shape[0]):
            trajectory3d[i] = trajectory3d[i] @ R[i].T + T[i]
    
    trajectory3d *= focallength_calibration

    # plot trajectory on the canvas
    trajectory2d = torch.div(trajectory3d, trajectory3d[..., 2:])
    trajectory2d = torch.matmul(K, trajectory2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]

    return trajectory2d


def world2incam(pred_ay_verts, R, T, cfg, focallength_calibration=1.0):
    # sometimes, fix the depth problem when the focal length estimation is incorrect
    pred_ay_verts[..., 2] *= cfg.adjust_focal
    
    if cfg.static_cam:
        pred_c_verts_edited = pred_ay_verts @ R[0].T + T[0] # world to incam
    else:
        pred_c_verts_edited = []
        for i in range(pred_ay_verts.shape[0]):
            pred_c_verts_edited.append(pred_ay_verts[i] @ R[i].T + T[i]) # world to incam
        pred_c_verts_edited = torch.stack(pred_c_verts_edited)

    pred_c_verts_edited *= focallength_calibration
    
    return pred_c_verts_edited


def incam2world(given_trajectory2d, R, T, K, depth, focallength_calibration=1.0):
    given_trajectory3d_incam = (torch.linalg.inv(K) @ points_padding(given_trajectory2d).T * depth).T
    given_trajectory3d_incam /= focallength_calibration # inverse calibration
    given_trajectory3d_global = (given_trajectory3d_incam - T) @ torch.linalg.inv(R).T

    return given_trajectory3d_global


def transform_smpl_params_incam_from_global(pred, cfg, R, T):
    if cfg.static_cam:
        pred["smpl_params_incam"]["global_orient"] = matrix_to_axis_angle(R[0].T @ axis_angle_to_matrix(pred["smpl_params_global"]["global_orient"]))
        pred["smpl_params_incam"]["transl"] = pred["smpl_params_global"]["transl"] @ R[0].T + T[0]
    else:
        for i in range(pred["smpl_params_incam"]["global_orient"].shape[0]):
            pred["smpl_params_incam"]["global_orient"][i] = matrix_to_axis_angle(R[i].T @ axis_angle_to_matrix(pred["smpl_params_global"]["global_orient"][i]))
            pred["smpl_params_incam"]["transl"][i] = pred["smpl_params_global"]["transl"][i] @ R[i].T + T[i]
    
    return pred


def align_speed_with_reference(points_ref, points, scale=1):
    from scipy.interpolate import interp1d

    def compute_cumulative_arc_length(points):
        deltas = torch.diff(points, dim=0)
        distances = torch.sqrt((deltas ** 2).sum(dim=1))
        arc_length = torch.cat([torch.zeros(1, device=points.device), distances.cumsum(dim=0)])
        return arc_length

    arc_length_ref = compute_cumulative_arc_length(points_ref)
    arc_length = compute_cumulative_arc_length(points) * scale
    reparameterized_points = torch.zeros((arc_length_ref.size(0), points.size(1)), dtype=points.dtype, device=points.device)

    for i in range(points.size(1)):
        interp_func = interp1d(arc_length.cpu().numpy(), points[:, i].cpu().numpy(), kind='linear', fill_value='extrapolate')
        reparameterized_points[:, i] = torch.tensor(interp_func(arc_length_ref.cpu().numpy()), dtype=points.dtype, device=points.device)
        # fix trajectory without extrapolation
        # reparameterized_points[:, i] = torch.tensor(interp_func(np.clip(arc_length_ref.cpu().numpy(), 0, arc_length[-1].cpu().numpy())), dtype=points.dtype, device=points.device) # fix distance
    
    return reparameterized_points


def moving_average(data, window_size=5):
    assert window_size % 2 == 1
    pad = window_size // 2
    data_padded = torch.nn.functional.pad(data.unsqueeze(0), (pad, pad), mode='replicate')
    
    kernel = torch.ones(window_size, device=data.device) / window_size
    smoothed_data = torch.conv1d(data_padded.view(1, 1, -1), kernel.view(1, 1, -1), padding=0).view(-1)
    
    return smoothed_data.squeeze(0)


def unwrap_angles(angles):
    # unwrap angle before using moving average
    unwrapped_angles = angles.clone()
    two_pi = 2 * np.pi

    for i in range(1, angles.size(0)):
        delta_angle = angles[i] - angles[i - 1]
        if delta_angle > np.pi:
            unwrapped_angles[i:] -= two_pi
        elif delta_angle < -np.pi:
            unwrapped_angles[i:] += two_pi

    return unwrapped_angles


def unwrap_axis_angles(angles):
    # unwrap angle before using moving average
    unwrap_axis_angles = angles.clone()

    reference_vector = unwrap_axis_angles[0, :]

    for i in range(1, angles.size(0)):
        current_vector = unwrap_axis_angles[i, :]

        dot_product = torch.dot(reference_vector, current_vector)

        if dot_product < 0:
            unwrap_axis_angles[i, :] = -current_vector  # Flip the sign

    return unwrap_axis_angles


def get_depth(depth_map, h, w, avg_depth, cfg):
    h, w = h.to(torch.long), w.to(torch.long)
    if max(h) < depth_map.shape[0] and max(w) < depth_map.shape[1]:
        depth = depth_map[h, w]
    else:
        depth = []
        for _h, _w in zip(h, w):
            if _h < depth_map.shape[0] and _w < depth_map.shape[1]:
                depth.append(depth_map[_h, _w])
            else:
                depth.append(avg_depth)
        depth = torch.cat(depth)
    
    return depth


def get_global_orient_from_trajectory3d(trajectory3d):
    diff = torch.nn.functional.pad((trajectory3d[1:] - trajectory3d[:-1]).unsqueeze(0), (0, 0, 0, 1), 'replicate').squeeze(0)
    angles = torch.atan2(diff[..., 0], diff[..., 2])

    return moving_average(unwrap_angles(angles), window_size=31)


def get_orient_transform(original_global_orient, given_trajectory3d_global):
    global_orient = unwrap_axis_angles(original_global_orient) # avoid interpolation error
    global_orient = axis_angle_to_quaternion(global_orient) # only interpolate using quaternion to avoid gimbal lock
    global_orient = torch.stack([moving_average(global_orient[:, i], window_size=31) for i in range(4)], -1)
    inverse_global_orient = torch.linalg.inv(quaternion_to_matrix(global_orient))
    given_rotate_matrix_xy = torch.stack([get_rotation_matrix_xz_plane(angle, device=global_orient.device) for angle in get_global_orient_from_trajectory3d(given_trajectory3d_global)])

    orient_transform = inverse_global_orient.transpose(1, 2) @ given_rotate_matrix_xy

    return orient_transform