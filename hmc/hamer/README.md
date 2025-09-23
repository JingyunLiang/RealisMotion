# Hamer Preparation

The hamer (hand pose) is optional, but it can help RealisMotion control the hand poses. Otherwise, we will use the standard hand pose in SMPL-X. To obtain the hamer from a video, please ref to [RealisDance](https://github.com/damo-cv/RealisDance/blob/RealisDance-SD/prepare_pose/README.md). Use the hacked `inference_video.py` in this folder to save the `hamer_info.pkl` files. Please set `OUTPUT_PATH` as the same folder as the GVHMR outputs. We will match it with the human movement, and render corresponding hamer condition videos during motion editing.


```
python inference_video.py --video_path {YOUR_VIDEO_PATH} --output_path {OUTPUT_PATH}
```