import numpy as np
import torch
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.renderer import (
    BlendParams,
    DirectionalLights,
    HardPhongShader,
    Materials,
    MeshRasterizer,
    MeshRendererWithFragments,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
)
from pytorch3d.renderer.cameras import look_at_rotation
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.transforms import axis_angle_to_matrix

from hmr4d.utils.vis.renderer_tools import checkerboard_geometry

colors_str_map = {
    "gray": [0.8, 0.8, 0.8],
    "green": [39, 194, 128],
}
import torch.nn as nn
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    softmax_rgb_blend,
)


class SimpleShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = (
            blend_params if blend_params is not None else BlendParams()
        )

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image


def overlay_image_onto_background(image, mask, bbox, background):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    out_image = background.copy()
    bbox = bbox[0].int().cpu().numpy().copy()
    roi_image = out_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    roi_image[mask] = image[mask]
    out_image[bbox[1] : bbox[3], bbox[0] : bbox[2]] = roi_image

    return out_image


def update_intrinsics_from_bbox(K_org, bbox):
    device, dtype = K_org.device, K_org.dtype

    K = torch.zeros((K_org.shape[0], 4, 4)).to(device=device, dtype=dtype)
    K[:, :3, :3] = K_org.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1

    image_sizes = []
    for idx, bbox in enumerate(bbox):
        left, upper, right, lower = bbox
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]

        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)
        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        K[idx, 0, 2] = new_cx
        K[idx, 1, 2] = new_cy
        image_sizes.append((int(new_height), int(new_width)))

    return K, image_sizes


def perspective_projection(x3d, K, R=None, T=None):
    if R != None:
        x3d = torch.matmul(R, x3d.transpose(1, 2)).transpose(1, 2)
    if T != None:
        x3d = x3d + T.transpose(1, 2)

    x2d = torch.div(x3d, x3d[..., 2:])
    x2d = torch.matmul(K, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    return x2d


def compute_bbox_from_points(X, img_w, img_h, scaleFactor=1.2):
    left = torch.clamp(X.min(1)[0][:, 0], min=0, max=img_w)
    right = torch.clamp(X.max(1)[0][:, 0], min=0, max=img_w)
    top = torch.clamp(X.min(1)[0][:, 1], min=0, max=img_h)
    bottom = torch.clamp(X.max(1)[0][:, 1], min=0, max=img_h)

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = right - left
    height = bottom - top

    new_left = torch.clamp(cx - width / 2 * scaleFactor, min=0, max=img_w - 1)
    new_right = torch.clamp(cx + width / 2 * scaleFactor, min=1, max=img_w)
    new_top = torch.clamp(cy - height / 2 * scaleFactor, min=0, max=img_h - 1)
    new_bottom = torch.clamp(cy + height / 2 * scaleFactor, min=1, max=img_h)

    bbox = (
        torch.stack(
            (
                new_left.detach(),
                new_top.detach(),
                new_right.detach(),
                new_bottom.detach(),
            )
        )
        .int()
        .float()
        .T
    )

    return bbox


class Renderer:
    def __init__(
        self,
        width,
        height,
        focal_length=None,
        device="cuda",
        faces=None,
        hand_faces=None,
        K=None,
        bin_size=None,
    ):
        """set bin_size to 0 for no binning"""
        self.width = width
        self.height = height
        self.bin_size = bin_size
        assert (focal_length is not None) ^ (
            K is not None
        ), "focal_length and K are mutually exclusive"

        self.device = device
        if faces is not None:
            if isinstance(faces, np.ndarray):
                faces = torch.from_numpy((faces).astype("int"))
            self.faces = faces.unsqueeze(0).to(self.device)
        if hand_faces is not None:
            if isinstance(hand_faces, np.ndarray):
                hand_faces = torch.from_numpy((hand_faces).astype("int"))
            self.hand_faces = hand_faces.unsqueeze(0).to(self.device)

        self.initialize_camera_params(focal_length, K)
        # self.lights = AmbientLights(
        #     ambient_color=((0.4, 0.4, 0.4),), device=device
        # )
        self.create_renderer()

    def create_renderer(self):

        self.blend_params = BlendParams(
            sigma=1e-4, gamma=1e-4, background_color=(0, 0, 0)
        )

        self.renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=self.image_sizes[0],
                    blur_radius=0,
                    bin_size=self.bin_size,
                ),
            ),
            shader=HardPhongShader(
                device=self.device, blend_params=self.blend_params
            ),
            # shader=SimpleShader(
            #     device=self.device, blend_params=self.blend_params
            # ),
        )

    def create_lights(self):
        light_direction = -self.R[
            :, :, 2
        ]  # The direction will be set to look in the same direction as the camera
        lights = DirectionalLights(
            direction=light_direction,  # The direction is set as the negative z-axis from camera's perspective
            ambient_color=torch.tensor([[0.4, 0.4, 0.4]]).to(
                self.device
            ),  # Ambient light color
            device=self.device,
        )
        return lights

    def create_lights_hamer(self, ambient_color=0.4, diffuse_color=0.3, specular_color=0.2):
        light_direction = -self.R[
            :, :, 2
        ]  # The direction will be set to look in the same direction as the camera
        lights = DirectionalLights(
            direction=light_direction,  # The direction is set as the negative z-axis from camera's perspective
            ambient_color=torch.tensor([[ambient_color, ambient_color, ambient_color]]).to(self.device),
            diffuse_color=torch.tensor([[diffuse_color, diffuse_color, diffuse_color]]).to(self.device),
            specular_color=torch.tensor([[specular_color, specular_color, specular_color]]).to(self.device),
            device=self.device,
        )
        return lights

    def create_camera(self, R=None, T=None):
        if R is not None:
            self.R = R.clone().view(1, 3, 3).to(self.device)
        if T is not None:
            self.T = T.clone().view(1, 3).to(self.device)

        return PerspectiveCameras(
            device=self.device,
            R=self.R.mT,
            T=self.T,
            K=self.K_full,
            image_size=self.image_sizes,
            in_ndc=False,
        )

    def initialize_camera_params(self, focal_length, K):
        # Extrinsics
        self.R = (
            torch.diag(torch.tensor([1, 1, 1]))
            .float()
            .to(self.device)
            .unsqueeze(0)
        )

        self.T = torch.tensor([0, 0, 0]).unsqueeze(0).float().to(self.device)

        # Intrinsics
        if K is not None:
            self.K = K.float().reshape(1, 3, 3).to(self.device)
        else:
            assert (
                focal_length is not None
            ), "focal_length or K should be provided"
            self.K = (
                torch.tensor(
                    [
                        [focal_length, 0, self.width / 2],
                        [0, focal_length, self.height / 2],
                        [0, 0, 1],
                    ]
                )
                .float()
                .reshape(1, 3, 3)
                .to(self.device)
            )
        self.bboxes = torch.tensor([[0, 0, self.width, self.height]]).float()
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(
            self.K, self.bboxes
        )
        self.cameras = self.create_camera()

    def set_intrinsic(self, K):
        self.K = K.reshape(1, 3, 3)

    def set_ground(self, length, center_x, center_z):
        device = self.device
        length, center_x, center_z = map(float, (length, center_x, center_z))
        v, f, vc, fc = map(
            torch.from_numpy,
            checkerboard_geometry(
                length=length, c1=center_x, c2=center_z, up="y"
            ),
        )
        v, f, vc = v.to(device), f.to(device), vc.to(device)
        self.ground_geometry = [v, f, vc]

    def update_bbox(self, x3d, scale=2.0, mask=None):
        """Update bbox of cameras from the given 3d points

        x3d: input 3D keypoints (or vertices), (num_frames, num_points, 3)
        """

        if x3d.size(-1) != 3:
            x2d = x3d.unsqueeze(0)
        else:
            x2d = perspective_projection(
                x3d.unsqueeze(0), self.K, self.R, self.T.reshape(1, 3, 1)
            )

        if mask is not None:
            x2d = x2d[:, ~mask]

        bbox = compute_bbox_from_points(x2d, self.width, self.height, scale)
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(
            self.K, bbox
        )
        self.cameras = self.create_camera()
        self.create_renderer()

    def reset_bbox(
        self,
    ):
        bbox = torch.zeros((1, 4)).float().to(self.device)
        bbox[0, 2] = self.width
        bbox[0, 3] = self.height
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(
            self.K, bbox
        )
        self.cameras = self.create_camera()
        self.create_renderer()

    def calculate_normals(self, meshes, fragments):
        verts = meshes.verts_packed()  # (V, 3)
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_verts = verts[faces]
        faces_normals = vertex_normals[
            faces
        ]  # if we want to render the blue normals...
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face,
            fragments.bary_coords,
            faces_normals,  # ones = torch.ones_like(fragments.bary_coords)
        )
        return pixel_normals

    def calculate_depth(self, fragments, pad_value=0):
        absolute_depth = fragments.zbuf[..., 0]
        no_depth = -1

        depth_min, depth_max = (
            absolute_depth[absolute_depth != no_depth].min(),
            absolute_depth[absolute_depth != no_depth].max(),
        )
        target_min, target_max = 50, 255

        depth_value = absolute_depth[absolute_depth != no_depth]
        depth_value = depth_max - depth_value

        depth_value /= depth_max - depth_min
        depth_value = depth_value * (target_max - target_min) + target_min

        relative_depth = absolute_depth.clone()
        relative_depth[absolute_depth != no_depth] = depth_value
        relative_depth[absolute_depth == no_depth] = (
            pad_value  # not completely black
        )
        return relative_depth

    def calculate_hamer(self, vertices, colors, materials, smpl_fragments):
        if isinstance(colors, torch.Tensor):
            verts_features = colors.unsqueeze(0).to(
                device=vertices.device
            )  # (num_frames, num_vertices, 3)
        else:
            if colors[0] > 1:
                colors = [c / 255.0 for c in colors]
            verts_features = (
                torch.tensor(colors)
                .reshape(1, 1, 3)
                .to(device=vertices.device, dtype=vertices.dtype)
            )
            verts_features = verts_features.repeat(1, vertices.shape[1], 1)
        textures = TexturesVertex(verts_features=verts_features)

        vertices = vertices.unsqueeze(0)
        
        mesh = Meshes(verts=vertices, faces=self.hand_faces, textures=textures)
        results, fragments = self.renderer(
            mesh,
            cameras=self.cameras,
            lights=self.create_lights_hamer(0.3, 0.75, 0.2),
            materials=materials,
        )
        image =  torch.clip(results[0, ..., :3], 0, 1) * 255
        hand_depth = fragments.zbuf.min(dim=-1)[0]  # [F,H,W]

        # 遮挡的hands mask掉，仅需要对smpl hand进行遮挡
        smpl_depth = smpl_fragments.zbuf.min(dim=-1)[0]  # [F,H,W]
        hand_mask = (hand_depth <= smpl_depth).float().squeeze(0).unsqueeze(-1)  # [H,W,1]
        image = image * hand_mask + torch.zeros_like(image) * (1 - hand_mask)

        image = torch.flip(image, [0, 1])

        return image

    def render_mesh(
        self, vertices, background=None, colors=[0.8, 0.8, 0.8], VI=50, hand_vertices=None, hand_colors=None,
    ):
        self.update_bbox(vertices[::VI], scale=1.2)

        if background is None:
            background = (
                np.zeros((self.height, self.width, 3)).astype(np.uint8) * 255
            )

        vertices = vertices.unsqueeze(0)

        if isinstance(colors, torch.Tensor):
            verts_features = colors.unsqueeze(0).to(
                device=vertices.device
            )  # (num_frames, num_vertices, 3)
        else:
            if colors[0] > 1:
                colors = [c / 255.0 for c in colors]
            verts_features = (
                torch.tensor(colors)
                .reshape(1, 1, 3)
                .to(device=vertices.device, dtype=vertices.dtype)
            )
            verts_features = verts_features.repeat(1, vertices.shape[1], 1)
        textures = TexturesVertex(verts_features=verts_features)

        mesh = Meshes(
            verts=vertices,
            faces=self.faces,
            textures=textures,
        )
        materials = Materials(device=self.device, shininess=0)
        results, fragments = self.renderer(
            mesh,
            cameras=self.cameras,
            lights=self.create_lights(),
            materials=materials,
        )
        results = torch.flip(
            results,
            [1, 2],
        )
        image = results[0, ..., :3] * 255
        mask = results[0, ..., -1] > 1e-3
        image = overlay_image_onto_background(
            image, mask, self.bboxes, background.copy()
        )
        # normals
        normals = self.calculate_normals(mesh, fragments)
        normal_map = softmax_rgb_blend(normals, fragments, self.blend_params)[
            ..., :4
        ]
        normal_map = torch.flip(normal_map, [1, 2])
        normal_map = ((normal_map[0, ..., :3] + 1) / 2) * 255
        mask = results[0, ..., -1] > 1e-3
        normal_map = overlay_image_onto_background(
            normal_map, mask, self.bboxes, background.copy()
        )
        # depths
        depths = self.calculate_depth(fragments)
        depths = depths.permute(1, 2, 0).repeat([1, 1, 3])
        depths = torch.flip(depths, [0, 1])
        mask = results[0, ..., -1] > 1e-3
        depths = overlay_image_onto_background(
            depths, mask, self.bboxes, background.copy()
        )
        # hamers
        if hand_vertices is not None:
            hamers = self.calculate_hamer(hand_vertices, hand_colors, materials, fragments)
            mask = results[0, ..., -1] > 1e-3
            hamers = overlay_image_onto_background(
                hamers, mask, self.bboxes, background.copy()
            )
        else:
            hamers = None

        self.reset_bbox()
        return image, normal_map, depths, hamers

    def render_with_ground(self, verts, colors, cameras, lights, faces=None):
        """
        :param verts (N, V, 3), potential multiple people
        :param colors (N, 3) or (N, V, 3)
        :param faces (N, F, 3), optional, otherwise self.faces is used will be used
        """
        # Sanity check of input verts, colors and faces: (B, V, 3), (B, F, 3), (B, V, 3)
        N, V, _ = verts.shape
        if faces is None:
            faces = self.faces.clone().expand(N, -1, -1)
        else:
            assert len(faces.shape) == 3, "faces should have shape of (N, F, 3)"

        assert len(colors.shape) in [2, 3]
        if len(colors.shape) == 2:
            assert len(colors) == N, "colors of shape 2 should be (N, 3)"
            colors = colors[:, None]
        colors = colors.expand(N, V, -1)[..., :3]

        # (V, 3), (F, 3), (V, 3)
        gv, gf, gc = self.ground_geometry
        verts = list(torch.unbind(verts, dim=0)) + [gv]
        faces = list(torch.unbind(faces, dim=0)) + [gf]
        colors = list(torch.unbind(colors, dim=0)) + [gc[..., :3]]
        mesh = create_meshes(verts, faces, colors)

        materials = Materials(device=self.device, shininess=0)

        results = self.renderer(
            mesh, cameras=cameras, lights=lights, materials=materials
        )
        image = (results[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)

        return image


def create_meshes(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (B, F, 3)
    :param colors (B, V, 3)
    """
    textures = TexturesVertex(verts_features=colors)
    meshes = Meshes(verts=verts, faces=faces, textures=textures)
    return join_meshes_as_scene(meshes)


def get_global_cameras(
    verts, device="cuda", distance=5, position=(-5.0, 5.0, 0.0)
):
    """This always put object at the center of view"""
    positions = torch.tensor([position]).repeat(len(verts), 1)
    targets = verts.mean(1)

    directions = targets - positions
    directions = (
        directions / torch.norm(directions, dim=-1).unsqueeze(-1) * distance
    )
    positions = targets - directions

    rotation = look_at_rotation(positions, targets).mT
    translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)

    lights = PointLights(device=device, location=[position])
    return rotation, translation, lights


def get_global_cameras_static(
    verts,
    beta=4.0,
    cam_height_degree=30,
    target_center_height=1.0,
    use_long_axis=False,
    vec_rot=45,
    device="cuda",
):
    L, V, _ = verts.shape

    # Compute target trajectory, denote as center + scale
    targets = verts.mean(1)  # (L, 3)
    targets[:, 1] = 0  # project to xz-plane
    target_center = targets.mean(0)  # (3,)
    target_scale, target_idx = torch.norm(targets - target_center, dim=-1).max(
        0
    )

    # a 45 degree vec from longest axis
    if use_long_axis:
        long_vec = targets[target_idx] - target_center  # (x, 0, z)
        long_vec = long_vec / torch.norm(long_vec)
        R = axis_angle_to_matrix(torch.tensor([0, np.pi / 4, 0])).to(long_vec)
        vec = R @ long_vec
    else:
        vec_rad = vec_rot / 180 * np.pi
        vec = torch.tensor([np.sin(vec_rad), 0, np.cos(vec_rad)]).float()
        vec = vec / torch.norm(vec)

    # Compute camera position (center + scale * vec * beta) + y=4
    target_scale = max(target_scale, 1.0) * beta
    position = target_center + vec * target_scale
    position[1] = (
        target_scale * np.tan(np.pi * cam_height_degree / 180)
        + target_center_height
    )

    # Compute camera rotation and translation
    positions = position.unsqueeze(0).repeat(L, 1)
    target_centers = target_center.unsqueeze(0).repeat(L, 1)
    target_centers[:, 1] = target_center_height
    rotation = look_at_rotation(positions, target_centers).mT
    translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)

    lights = PointLights(device=device, location=[position.tolist()])
    return rotation, translation, lights


def get_ground_params_from_points(root_points, vert_points):
    """xz-plane is the ground plane
    Args:
        root_points: (L, 3), to decide center
        vert_points: (L, V, 3), to decide scale
    """
    root_max = root_points.max(0)[0]  # (3,)
    root_min = root_points.min(0)[0]  # (3,)
    cx, _, cz = (root_max + root_min) / 2.0

    vert_max = vert_points.reshape(-1, 3).max(0)[0]  # (L, 3)
    vert_min = vert_points.reshape(-1, 3).min(0)[0]  # (L, 3)
    scale = (vert_max - vert_min)[[0, 2]].max()
    return float(scale), float(cx), float(cz)
