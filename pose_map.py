#!/usr/bin/env python3
"""
YOLO Pose + Depth-Anything-V2 joint depth estimation
Tested with: Depth-Anything-V2-Small (vits, 24.8M params)


Usage:
   python3 pose_detection_depth.py \
       --source "http://10.203.184.12:8080/stream.mjpg" \
       --model yolo11n-pose.pt \
       --depth-model /home/zmao16/Downloads/depth_anything_v2_vits.pth \
       --encoder vits \
       --show-depth
"""


import cv2
import argparse
import numpy as np
import torch
from ultralytics import YOLO
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2


# COCO 17 keypoint names (YOLO pose order)
KEYPOINT_NAMES = [
   "nose", "eye_L", "eye_R", "ear_L", "ear_R",
   "shl_L", "shl_R", "elb_L", "elb_R", "wri_L", "wri_R",
   "hip_L", "hip_R", "kne_L", "kne_R", "ank_L", "ank_R",
]


MODEL_CONFIGS = {
   "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192,  384]},
   "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384,  768]},
   "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
   "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536,1536,1536,1536]},
}




def load_depth_model(ckpt_path: str, encoder: str, device: str):
   cfg = MODEL_CONFIGS[encoder]
   model = DepthAnythingV2(**cfg)
   model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
   model = model.to(device).eval()
   print(f"[Depth] Loaded {encoder} checkpoint: {ckpt_path}  (device={device})")
   return model




def depth_colormap(depth_map: np.ndarray) -> np.ndarray:
   """Normalize depth → uint8 → apply COLORMAP_INFERNO (near=bright)."""
   d_min, d_max = depth_map.min(), depth_map.max()
   if d_max - d_min < 1e-6:
       norm = np.zeros_like(depth_map, dtype=np.uint8)
   else:
       norm = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
   # Invert so closer = higher value → warmer color
   norm = 255 - norm
   return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)




def depth_to_color(depth_val: float, d_min: float, d_max: float):
   """Return BGR color for a single depth value (near=red, far=blue)."""
   if d_max - d_min < 1e-6:
       t = 0.5
   else:
       t = 1.0 - (depth_val - d_min) / (d_max - d_min)  # 1=near, 0=far
   # Interpolate: far=blue(255,0,0) → near=red(0,0,255)
   b = int((1 - t) * 255)
   r = int(t * 255)
   return (b, 0, r)




def draw_joints_with_depth(
   frame: np.ndarray,
   keypoints_xy: np.ndarray,   # (17, 2) float
   keypoints_conf: np.ndarray, # (17,)   float
   depth_map: np.ndarray,      # (H, W)  float
   conf_thr: float = 0.3,
   show_name: bool = False,
):
   """Overlay joint depth values onto the frame."""
   h, w = depth_map.shape
   fh, fw = frame.shape[:2]
   d_min, d_max = depth_map.min(), depth_map.max()


   for idx, (xy, conf) in enumerate(zip(keypoints_xy, keypoints_conf)):
       if conf < conf_thr:
           continue
       px = int(np.clip(xy[0], 0, fw - 1))
       py = int(np.clip(xy[1], 0, fh - 1))


       # Sample depth at joint pixel (map frame coords → depth map coords)
       dx = int(px * w / fw)
       dy = int(py * h / fh)
       dx = int(np.clip(dx, 0, w - 1))
       dy = int(np.clip(dy, 0, h - 1))
       d_val = depth_map[dy, dx]


       color = depth_to_color(d_val, d_min, d_max)


       # Draw circle at joint
       cv2.circle(frame, (px, py), 6, color, -1)
       cv2.circle(frame, (px, py), 6, (255, 255, 255), 1)  # white outline


       # Depth label
       label = f"{d_val:.2f}"
       if show_name:
           label = f"{KEYPOINT_NAMES[idx]}:{d_val:.2f}"
       cv2.putText(
           frame, label,
           (px + 8, py - 4),
           cv2.FONT_HERSHEY_SIMPLEX, 0.38,
           color, 1, cv2.LINE_AA,
       )




def open_camera(source: str):
   cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
   if not cap.isOpened():
       raise RuntimeError(f"Could not open camera source: {source}")
   return cap




def main():
   parser = argparse.ArgumentParser(description="YOLO Pose + Depth-Anything-V2 per-joint depth")
   parser.add_argument("--source",      type=str,   default="0")
   parser.add_argument("--model",       type=str,   default="yolo11n-pose.pt")
   parser.add_argument("--conf",        type=float, default=0.5)
   parser.add_argument("--width",       type=int,   default=1280)
   parser.add_argument("--height",      type=int,   default=720)
   # Depth-Anything-V2 args
   parser.add_argument("--depth-model", type=str,   default="checkpoints/depth_anything_v2_vits.pth",
                       help="Path to Depth-Anything-V2 .pth checkpoint")
   parser.add_argument("--encoder",     type=str,   default="vits",
                       choices=["vits", "vitb", "vitl", "vitg"],
                       help="Encoder variant matching your checkpoint")
   parser.add_argument("--input-size",  type=int,   default=518,
                       help="Depth model input size (default 518)")
   parser.add_argument("--depth-every", type=int,   default=1,
                       help="Run depth inference every N frames (1=every frame, 2=every other...)")
   parser.add_argument("--show-depth",  action="store_true",
                       help="Show a second window with the depth colormap")
   parser.add_argument("--show-names",  action="store_true",
                       help="Show keypoint names next to depth values")
   parser.add_argument("--kp-conf",     type=float, default=0.3,
                       help="Min keypoint confidence to display depth")
   args = parser.parse_args()


   # ── Device ────────────────────────────────────────────────────────────────
   DEVICE = (
       "cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available()
       else "cpu"
   )
   print(f"[Device] {DEVICE}")


   # ── Load models ───────────────────────────────────────────────────────────
   model_pose  = YOLO(args.model)
   model_depth = load_depth_model(args.depth_model, args.encoder, DEVICE)


   # ── Camera ────────────────────────────────────────────────────────────────
   cap = open_camera(args.source)
   cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)


   cv2.namedWindow("Pose + Depth", cv2.WINDOW_NORMAL)
   if args.show_depth:
       cv2.namedWindow("Depth Map",   cv2.WINDOW_NORMAL)


   depth_map    = None   # cached depth (H×W float)
   frame_count  = 0


   while True:
       ret, frame = cap.read()
       if not ret:
           print("Failed to read frame.")
           break


       frame_count += 1


       # ── Depth inference (every N frames) ──────────────────────────────────
       if frame_count % args.depth_every == 0:
           with torch.no_grad():
               depth_map = model_depth.infer_image(frame, args.input_size)
               # depth_map: H×W numpy float32, larger=farther (relative)


       # ── YOLO Pose inference ───────────────────────────────────────────────
       results = model_pose.predict(source=frame, conf=args.conf, verbose=False)
       result  = results[0]


       # Draw skeleton (YOLO built-in)
       display = result.plot()


       # ── Per-joint depth overlay ───────────────────────────────────────────
       if depth_map is not None and result.keypoints is not None:
           kp_data = result.keypoints  # ultralytics Keypoints object
           # xy: (N_persons, 17, 2), conf: (N_persons, 17)
           xy_all   = kp_data.xy.cpu().numpy()    # pixel coords in orig frame
           conf_all = kp_data.conf.cpu().numpy()  # confidence scores


           for person_xy, person_conf in zip(xy_all, conf_all):
               draw_joints_with_depth(
                   display,
                   person_xy, person_conf,
                   depth_map,
                   conf_thr=args.kp_conf,
                   show_name=args.show_names,
               )


       # ── HUD ───────────────────────────────────────────────────────────────
       cv2.putText(display, f"Source: {args.source}",
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
       cv2.putText(display, f"Depth encoder: {args.encoder}",
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(200,255,200), 1)
       cv2.putText(display, "Press q to quit",
                   (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(255,255,255), 1)


       cv2.imshow("Pose + Depth", display)


       # ── Depth colormap window (optional) ─────────────────────────────────
       if args.show_depth and depth_map is not None:
           depth_vis = depth_colormap(depth_map)
           depth_vis_resized = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]))


           # Draw keypoint dots on depth map too
           if result.keypoints is not None:
               xy_all   = result.keypoints.xy.cpu().numpy()
               conf_all = result.keypoints.conf.cpu().numpy()
               dh, dw = depth_map.shape
               fh, fw = frame.shape[:2]
               for person_xy, person_conf in zip(xy_all, conf_all):
                   for xy, conf in zip(person_xy, person_conf):
                       if conf < args.kp_conf:
                           continue
                       px = int(np.clip(xy[0], 0, fw-1))
                       py = int(np.clip(xy[1], 0, fh-1))
                       cv2.circle(depth_vis_resized, (px, py), 5, (0,255,0), -1)
           cv2.imshow("Depth Map", depth_vis_resized)


       if cv2.waitKey(1) & 0xFF == ord("q"):
           break


   cap.release()
   cv2.destroyAllWindows()




if __name__ == "__main__":
   main()