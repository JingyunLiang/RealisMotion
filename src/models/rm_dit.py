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
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import ModelMixin, CacheMixin
from diffusers.configuration_utils import register_to_config, ConfigMixin
from diffusers.loaders import PeftAdapterMixin, FromOriginalModelMixin
from diffusers.models.attention_processor import Attention
from diffusers.models.controlnet import zero_module
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import (
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    WanTransformerBlock,
    WanAttnProcessor2_0,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
    BaseOutput,
)

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from .attention import flash_attention


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ShiftedWanRotaryPosEmbed(WanRotaryPosEmbed):

    def shifted_forward(self, hidden_states: torch.Tensor, grid_sizes: List[torch.Tensor]) -> torch.Tensor:
        
        self.freqs = self.freqs.to(hidden_states.device)
        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        # we shift a sufficiently large value 80
        shifts = [grid_sizes[-1][0] + 80, 0, 0]
        freqs_list = []
        for i, grid_size in enumerate(grid_sizes):
            f, h, w = grid_size

            # do not shift for the video
            if i == len(grid_sizes) - 1:
                shifts = [0, 0, 0]

            freqs_f = freqs[0][shifts[0]: shifts[0] + f].view(f, 1, 1, -1).expand(f, h, w, -1)
            freqs_h = freqs[1][shifts[1]: shifts[1] + h].view(1, h, 1, -1).expand(f, h, w, -1)
            freqs_w = freqs[2][shifts[2]: shifts[2] + w].view(1, 1, w, -1).expand(f, h, w, -1)
            freqs_i = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, f * h * w, -1)
            
            freqs_list.append(freqs_i)
            shifts[0] = shifts[0] + 1

        freqs = torch.cat(freqs_list, dim=2)
        
        return freqs


class SelfAttnProcessorSP(WanAttnProcessor2_0):
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        seq_lens: Optional[int] = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        if get_sequence_parallel_world_size() > 1:
            # convert [batch, num_head, length, channel] -> [batch, length, num_head, channel]
            query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

            # convert to half
            def half(x):
                return x if x.dtype in (torch.float16, torch.bfloat16) else x.to(torch.bfloat16)
            original_dtype = query.dtype
            query, key, value = half(query), half(key), half(value)

            # do attention
            hidden_states = xFuserLongContextAttention()(
                None, query=query, key=key, value=value
            )
            # convert back
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(original_dtype)
        else:
            hidden_states = flash_attention(
                query, key, value, k_lens=seq_lens
            )
            hidden_states = hidden_states.flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class HackedWanTransformerBlock(WanTransformerBlock):

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        seq_lens: Optional[int] = None,
        context_lens: Optional[int] = None,
        ref_seq_len: Optional[int] = 0,
        padding_num: Optional[int] = 0,
        sp_degree:  Optional[int] = 1,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb
        ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, seq_lens=seq_lens)
        hidden_states = (hidden_states + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        
        # we do not apply cross attention for ref tokens and padded tokens
        if sp_degree > 1:
            attn_output = get_sp_group().all_gather(attn_output, dim=1)
            
            attn_output[:, :ref_seq_len] *= 0
            if padding_num > 0:
                attn_output[:, -padding_num:] *= 0

            attn_output = torch.chunk(attn_output, sp_degree, dim=1)[get_sequence_parallel_rank()]
        else:
            attn_output[:, :ref_seq_len] *= 0
            if padding_num > 0:
                attn_output[:, -padding_num:] *= 0

        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states + ff_output * c_gate_msa).type_as(hidden_states)

        return hidden_states


@dataclass
class RealisMotionOutput(BaseOutput):
    sample: "torch.Tensor"
    teacache_kwargs: Optional[Dict[str, Any]] = None


class RealisMotion(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
        A Transformer model for video-like data used in the RealisMotion model (based on Wan-2.1 T2V).

        Args:
            patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
            num_attention_heads (`int`, defaults to `40`):
                Fixed length for text embeddings.
            attention_head_dim (`int`, defaults to `128`):
                The number of channels in each head.
            in_channels (`int`, defaults to `16`):
                The number of channels in the input.
            out_channels (`int`, defaults to `16`):
                The number of channels in the output.
            text_dim (`int`, defaults to `512`):
                Input dimension for text embeddings.
            freq_dim (`int`, defaults to `256`):
                Dimension for sinusoidal time embeddings.
            ffn_dim (`int`, defaults to `13824`):
                Intermediate dimension in feed-forward network.
            num_layers (`int`, defaults to `40`):
                The number of layers of transformer blocks to use.
            window_size (`Tuple[int]`, defaults to `(-1, -1)`):
                Window size for local attention (-1 indicates global attention).
            cross_attn_norm (`bool`, defaults to `True`):
                Enable cross-attention normalization.
            qk_norm (`bool`, defaults to `True`):
                Enable query/key normalization.
            eps (`float`, defaults to `1e-6`):
                Epsilon value for normalization layers.
            add_img_emb (`bool`, defaults to `False`):
                Whether to use img_emb.
            added_kv_proj_dim (`int`, *optional*, defaults to `None`):
                The number of channels to use for the added key and value projections. If `None`, no projection is used.

            # Adiitional Args for RealisMotion
            ref_in_channels (`int`, defaults to 16):
                The number of channels in the reference input.
            controlnet_in_channels (`int`, defaults to 32):
                The number of channels in the ControlNet input.
            controlnet_shared_block (`int`, defaults to 5):
                Share the ControlNet layers every n Wan layers.
            controlnet_weights (`float`, defaults to 1.0):
                Weight for ControlNet addition.
        """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["HackedWanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 33,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        ref_in_channels: int = 32,
        controlnet_in_channels: int = 80,
        controlnet_shared_block: int = 5,
        controlnet_weights: float = 1.0,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = ShiftedWanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                HackedWanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # for foreground reference
        self.ref_patch_embedding = nn.Conv3d(ref_in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.ref_id_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.ref_id_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)

        # for ControlNet
        self.controlnet_patch_embedding = nn.Conv3d(controlnet_in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.controlnet_condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        self.controlnet_shared_block = controlnet_shared_block
        self.controlnet_weights = controlnet_weights
        self.controlnet_blocks = nn.ModuleList(
            [
                HackedWanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers // controlnet_shared_block)
            ]
        )
        self.controlnet_layers = nn.ModuleList(
            [
                zero_module(nn.Linear(inner_dim, inner_dim)) for _ in range(num_layers // controlnet_shared_block)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim ** 0.5)

        self.gradient_checkpointing = False
        self.sp_degree = 1

    def set_sp_degree(self, sp_degree: int):
        self.sp_degree = int(sp_degree)

        # We only modify self-attention here, as modifying cross-attention would require more 
        # implementation effort with limited additional speed gains.
        # use FlashAttention3 > FlashAttention2 > PyTorchAttention_2_0
        for block in self.blocks:
            block.attn1.set_processor(SelfAttnProcessorSP())
        for block in self.controlnet_blocks:
            block.attn1.set_processor(SelfAttnProcessorSP())


    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        ref_list: Optional[List[torch.Tensor]] = None,
        background: Optional[torch.Tensor] = None,
        motion: Optional[torch.Tensor] = None,
        enable_teacache: bool = False,
        current_step: int = 0,
        teacache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        
        # We move ControlNet to the end because we do not need to run it sometimes if teacache is enabled.
        # but we save the variables here for later usage.
        controlnet_hidden_states = hidden_states
        controlnet_encoder_hidden_states = encoder_hidden_states
        controlnet_encoder_hidden_states_image = encoder_hidden_states_image

        # ref embeddings
        ref_list = [self.ref_patch_embedding(_ref) for _ref in ref_list]
        grid_sizes = [torch.tensor(ref.shape[2:], dtype=torch.long) for ref in ref_list]
        ref_id = [torch.tensor([index  * 1000], device=hidden_states.device).repeat(batch_size) for index in range(1, len(ref_list) + 1)]
        ref_id = [self.ref_id_embedder(self.ref_id_proj(index).to(self.dtype)) for index in ref_id] # (N, D)
        refs = torch.cat([ref.flatten(2).transpose(1, 2) + _ref_id.unsqueeze(1) for ref, _ref_id in zip(ref_list, ref_id)], dim=1)
        ref_seq_len = refs.shape[1]

        # embeddings
        hidden_states = self.patch_embedding(torch.cat((hidden_states, background), 1))
        grid_sizes.append(torch.tensor(hidden_states.shape[2:], dtype=torch.long))
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        
        seq_lens = torch.tensor([u.size(0) for u in hidden_states], dtype=torch.long)
        context_lens = None
        
        rotary_emb = self.rope.shifted_forward(hidden_states, grid_sizes)
        hidden_states = torch.cat((refs, hidden_states), dim=1) # order [refs, hidden_states]

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        # for sp split
        padding_num = 0
        if self.sp_degree > 1:
            original_seq_len = hidden_states.shape[1]
            if original_seq_len % self.sp_degree != 0:
                # TODO: We should use attention mask to prevent processing padding tokens.
                # TODO: But currently, xFuserLongContextAttention does not support attention mask.
                padding_num = self.sp_degree - original_seq_len % self.sp_degree
                hidden_states = torch.cat(
                    [hidden_states, hidden_states.new_zeros(
                        hidden_states.shape[0], padding_num, hidden_states.shape[2])], dim=1)
                rotary_emb = torch.cat(
                    [rotary_emb, rotary_emb.new_zeros(
                        rotary_emb.shape[0], rotary_emb.shape[1], padding_num, rotary_emb.shape[-1])], dim=2)
            hidden_states = torch.chunk(hidden_states, self.sp_degree, dim=1)[get_sequence_parallel_rank()]
            rotary_emb = torch.chunk(rotary_emb, self.sp_degree, dim=2)[get_sequence_parallel_rank()]

        def _block_forward(x):
            # split the controlnet outputs as well
            if self.sp_degree > 1:
                for i in range(len(control_signals)):
                    control_signals[i] = torch.chunk(
                        torch.nn.functional.pad(control_signals[i], (0, 0, ref_seq_len, padding_num)),
                        self.sp_degree, dim=1)[get_sequence_parallel_rank()]

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                for i, block in enumerate(self.blocks):
                    x = self._gradient_checkpointing_func(
                        block, x, encoder_hidden_states, timestep_proj, rotary_emb, seq_lens, context_lens, ref_seq_len, padding_num, self.sp_degree
                    )
                    if self.sp_degree > 1:
                        x = x + control_signals[i // self.controlnet_shared_block] * self.controlnet_weights
                    else:
                        x[:, ref_seq_len:] = x[:, ref_seq_len:] + control_signals[i // self.controlnet_shared_block] * self.controlnet_weights
            else:
                for i, block in enumerate(self.blocks):
                    x = block(x, encoder_hidden_states, timestep_proj, rotary_emb, seq_lens, context_lens, ref_seq_len, padding_num, self.sp_degree)
                    if self.sp_degree > 1:
                        x = x + control_signals[i // self.controlnet_shared_block] * self.controlnet_weights
                    else:
                        x[:, ref_seq_len:] = x[:, ref_seq_len:] + control_signals[i // self.controlnet_shared_block] * self.controlnet_weights
            return x

        if enable_teacache:
            modulated_inp = timestep_proj if teacache_kwargs["use_timestep_proj"] else temb
            if (
                teacache_kwargs["previous_e0"] is None or
                teacache_kwargs["previous_residual"] is None or
                current_step < teacache_kwargs["ret_steps"] or
                current_step >= teacache_kwargs["cutoff_steps"]
            ):
                should_calc = True
            else:
                rescale_func = np.poly1d(teacache_kwargs["coefficients"])
                teacache_kwargs["accumulated_rel_l1_distance"] += rescale_func(
                    (
                        (modulated_inp - teacache_kwargs["previous_e0"]).abs().mean() /
                        teacache_kwargs["previous_e0"].abs().mean()
                    ).cpu().item()
                )
                if teacache_kwargs["accumulated_rel_l1_distance"] < teacache_kwargs["teacache_thresh"]:
                    should_calc = False
                else:
                    should_calc = True
                    teacache_kwargs["accumulated_rel_l1_distance"] = 0
            teacache_kwargs["previous_e0"] = modulated_inp.clone()
            if should_calc:
                ori_hidden_states = hidden_states.clone()
                control_signals = self.controlnet_forward(
                    hidden_states=controlnet_hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=controlnet_encoder_hidden_states,
                    encoder_hidden_states_image=controlnet_encoder_hidden_states_image,
                    motion=motion,
                ) 
                hidden_states = _block_forward(hidden_states)
                teacache_kwargs["previous_residual"] = hidden_states - ori_hidden_states
            else:
                hidden_states = hidden_states + teacache_kwargs["previous_residual"]
        else:
            control_signals = self.controlnet_forward(
                hidden_states=controlnet_hidden_states,
                timestep=timestep,
                encoder_hidden_states=controlnet_encoder_hidden_states,
                encoder_hidden_states_image=controlnet_encoder_hidden_states_image,
                motion=motion,
            ) 
            hidden_states = _block_forward(hidden_states)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        # for sp gather
        if self.sp_degree > 1:
            hidden_states = get_sp_group().all_gather(hidden_states, dim=1)
            if padding_num == 0:
                hidden_states = hidden_states[:, ref_seq_len:]
            else:
                hidden_states = hidden_states[:, ref_seq_len: -padding_num]
        else:
            hidden_states = hidden_states[:, ref_seq_len:]
        
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, teacache_kwargs,)

        return RealisMotionOutput(sample=output, teacache_kwargs=teacache_kwargs)

    def controlnet_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        motion: Optional[List[torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        hidden_states = torch.cat([hidden_states, motion], dim=1)

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.controlnet_patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        seq_lens = torch.tensor([u.size(0) for u in hidden_states], dtype=torch.long)
        context_lens = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.controlnet_condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        # for sp split
        if self.sp_degree > 1:
            original_seq_len = hidden_states.shape[1]
            padding_num = 0
            if original_seq_len % self.sp_degree != 0:
                # TODO: We should use attention mask to prevent processing padding tokens.
                # TODO: But currently, xFuserLongContextAttention does not support attention mask.
                padding_num = self.sp_degree - original_seq_len % self.sp_degree
                hidden_states = torch.cat(
                    [hidden_states, hidden_states.new_zeros(
                        hidden_states.shape[0], padding_num, hidden_states.shape[2])], dim=1)
                rotary_emb = torch.cat(
                    [rotary_emb, rotary_emb.new_zeros(
                        rotary_emb.shape[0], rotary_emb.shape[1], padding_num, rotary_emb.shape[-1])], dim=2)
            hidden_states = torch.chunk(hidden_states, self.sp_degree, dim=1)[get_sequence_parallel_rank()]
            rotary_emb = torch.chunk(rotary_emb, self.sp_degree, dim=2)[get_sequence_parallel_rank()]

        control_signals = []
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for i, block in enumerate(self.controlnet_blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, seq_lens, context_lens
                )
                control_signal = self._gradient_checkpointing_func(
                    self.controlnet_layers[i], hidden_states)
                
                if self.sp_degree > 1:
                    control_signal = get_sp_group().all_gather(control_signal, dim=1)
                    if padding_num > 0:
                        control_signal = control_signal[:, :-padding_num]
                
                control_signals.append(control_signal)
        else:
            for i, block in enumerate(self.controlnet_blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, seq_lens, context_lens)
                control_signal = self.controlnet_layers[i](hidden_states)
                
                if self.sp_degree > 1:
                    control_signal = get_sp_group().all_gather(control_signal, dim=1)
                    if padding_num > 0:
                        control_signal = control_signal[:, :-padding_num]
                    
                control_signals.append(control_signal)
        
        return control_signals

        
if __name__ == "__main__":
    # debug

    model=RealisMotion(num_layers=5).cuda()
    
    hidden_states = torch.rand(1,16,16,16,16).cuda()
    timestep = torch.rand(1).cuda()
    encoder_hidden_states = torch.rand(1, 512, 4096).cuda()
    background = torch.rand(1,17,16,16,16).cuda()
    ref_list = [torch.rand(1,32,1,16,16).cuda(), torch.rand(1,32,1,16,16).cuda()]
    motion = torch.rand(1,64,16,16,16).cuda()

    model(hidden_states,timestep,encoder_hidden_states,ref_list=ref_list,background=background,motion=motion)