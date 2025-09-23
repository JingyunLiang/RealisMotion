# More Editing Examples

```
export PYTHONPATH="YOUR_PATH/GVHMR/hmr4d:$PYTHONPATH"

python hmc/realismotion_render_demo.py \
    --video=inputs/example_video/internalaffairs.mp4 \
    --motion_path inputs/motion_bank/falldown2  \
    --reference_path inputs/demo/internalaffairs  \
    --output_root inputs/demo \
    --window_size 1 \
    --repeat_smpl 0 46 1 \
    --pause_at_begin 50 \
    --pause_at_end 105 \
    --edit_type affine_transform \
    --affine_transform_args 0 0 0 -0.3


python hmc/realismotion_render_demo.py \
    --video=inputs/example_video/couple1.mp4 \
    --motion_path inputs/motion_bank/goodbye_flip  \
    --reference_path inputs/demo/couple1  \
    --output_root inputs/demo \
    --window_size 1 \
    --repeat_smpl 40 -1 1 \
    --pause_at_begin 0 \
    --edit_type affine_transform \
    --affine_transform_args 0.95  0.7 1 0.1 \
    -s


python hmc/realismotion_render_demo.py \
    --video=inputs/example_video/couple2.mp4 \
    --motion_path inputs/motion_bank/love2  \
    --reference_path inputs/demo/couple2  \
    --output_root inputs/demo \
    --window_size 1 \
    --pause_at_begin 0 \
    --edit_type affine_transform \
    --affine_transform_args 0.95  0.7 1 0.05 \
    -s


python hmc/realismotion_render_demo.py \
    --video=inputs/example_video/couple2.mp4 \
    --motion_path inputs/motion_bank/comic  \
    --reference_path inputs/demo/couple2  \
    --output_root inputs/demo \
    --window_size 1 \
    --pause_at_begin 0 \
    --edit_type affine_transform \
    --affine_transform_args  0.95  0.8 1 0.05 \
    -s


python hmc/realismotion_render_demo.py \
    --video=inputs/example_video/justin.mp4 \
    --motion_path inputs/motion_bank/love1  \
    --reference_path inputs/demo/justin  \
    --output_root inputs/demo \
    --window_size 25 \
    --repeat_smpl 0 400 1 \
    --pause_at_begin 0 \
    --edit_type affine_transform \
    --affine_transform_args  0.8 0.5 2 0 \
    -s


python hmc/realismotion_render_demo.py \
    --video=inputs/example_video/taylor.mp4 \
    --motion_path inputs/motion_bank/tstagerobot  \
    --reference_path inputs/demo/taylor  \
    --output_root inputs/demo \
    --window_size 25 \
    --repeat_smpl 0 400 1 \
    --pause_at_begin 0 \
    --edit_type affine_transform \
    --affine_transform_args  0.8 0.6 1.4 -1.45 \
    -s


python hmc/realismotion_render_demo.py \
    --video=inputs/example_video/taylor.mp4 \
    --motion_path inputs/motion_bank/tstagegirl  \
    --reference_path inputs/demo/taylor  \
    --output_root inputs/demo \
    --window_size 25 \
    --repeat_smpl 2 31 12 \
    --pause_at_begin 0 \
    --edit_type affine_transform \
    --affine_transform_args  0.8 0.6 2.4 0 \
    -s


python hmc/realismotion_render_demo.py \
    --video=inputs/example_video/justin.mp4 \
    --motion_path inputs/motion_bank/end  \
    --reference_path inputs/demo/justin  \
    --output_root inputs/demo \
    --window_size 25 \
    --repeat_smpl 0 400 1 \
    --pause_at_begin 0 \
    --edit_type affine_transform \
    --affine_transform_args  0.8 0.4 1.1 0 \
    -s


python hmc/realismotion_render_demo.py \
    --video=inputs/example_video/male.mp4 \
    --motion_path inputs/motion_bank/jog  \
    --reference_path inputs/demo/male  \
    --output_root inputs/demo \
    --window_size 1 \
    --repeat_smpl 37 57 12 \
    --pause_at_begin 0 \
    --edit_type edit_trajectory \
    --edit_trajectory_args  1107 48 1098 662 1347 486 1154 303 952 682 1136 832 1100 494 910 787 \
    --append circle \
    -s


python hmc/realismotion_render_demo.py \
    --video=inputs/example_video/trumpmusk.mp4 \
    --motion_path inputs/motion_bank/ycma  \
    --reference_path inputs/demo/trumpmusk  \
    --output_root inputs/demo \
    --window_size 1 \
    --pause_at_begin 0 \
    --edit_type affine_transform \
    --affine_transform_args  0 0 0 0 \
    -s

    ```


