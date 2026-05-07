#!/usr/bin/env python3
"""
pose_detection_metric_depth.py

YOLO Pose + Depth-Anything-V2 + metric depth scaling from person bounding boxes.

Core idea:
    Z_metric ≈ fy * H_real / h_px

We use YOLO person detections to estimate a scale factor that converts the
relative depth map from Depth Anything V2 into approximate metric depth.

Example:
    python3 pose_detection_metric_depth.py \
        --source "http://10.203.184.12:8080/stream.mjpg" \
        --model yolo11n-pose.pt \
        --depth-model /home/cjing5/Downloads/depth_anything_v2_vits.pth \
        --encoder vits \
        --show-depth \
        --show-names \
        --person-height 1.70 \
        --fx 1320.594 --fy 1320.594 --cx 958.032 --cy 721.8218
"""

import cv2
import argparse
import numpy as np
import torch
import yaml

from dataclasses import dataclass
from typing import Optional
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
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_yaml(cls, path: str) -> "CameraIntrinsics":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls(
            fx=float(cfg["fx"]),
            fy=float(cfg["fy"]),
            cx=float(cfg["cx"]),
            cy=float(cfg["cy"]),
        )


class MetricDepthScaler:
    """
    Convert relative depth to approximate metric depth using person bounding boxes.

    Z_metric = fy * H_real / h_px
    scale = Z_metric / d_rel
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        person_height_m: float = 1.70,
        ema_alpha: float = 0.15,
        min_bbox_height: int = 40,
        min_conf: float = 0.5,
    ):
        self.K = intrinsics
        self.H_real = person_height_m
        self.alpha = ema_alpha
        self.min_bbox_h = min_bbox_height
        self.min_conf = min_conf
        self._scale_ema: Optional[float] = None

    def update(
        self,
        depth_map: np.ndarray,
        boxes_xyxy: np.ndarray,
        boxes_conf: np.ndarray,
        frame_shape: tuple,
    ) -> Optional[float]:
        scales = self._estimate_scales(depth_map, boxes_xyxy, boxes_conf, frame_shape)
        if not scales:
            return self._scale_ema

        raw_scale = float(np.median(scales))

        if self._scale_ema is None:
            self._scale_ema = raw_scale
        else:
            self._scale_ema = self.alpha * raw_scale + (1.0 - self.alpha) * self._scale_ema

        return self._scale_ema

    def to_metric(self, depth_map: np.ndarray) -> Optional[np.ndarray]:
        if self._scale_ema is None:
            return None
        return depth_map * self._scale_ema

    @property
    def scale(self) -> Optional[float]:
        return self._scale_ema

    def _estimate_scales(self, depth_map, boxes_xyxy, boxes_conf, frame_shape):
        fH, fW = frame_shape[:2]
        dH, dW = depth_map.shape
        scales = []

        for box, conf in zip(boxes_xyxy, boxes_conf):
            if conf < self.min_conf:
                continue

            x1, y1, x2, y2 = box.astype(float)
            h_px = y2 - y1
            if h_px < self.min_bbox_h:
                continue

            # Approximate metric depth using pinhole camera model
            Z_metric = self.K.fy * self.H_real / max(h_px, 1e-6)

            # Sample relative depth around bbox centroid
            cx = int(np.clip((x1 + x2) / 2.0, 0, fW - 1))
            cy = int(np.clip((y1 + y2) / 2.0, 0, fH - 1))

            dx = int(np.clip(cx * dW / fW, 0, dW - 1))
            dy = int(np.clip(cy * dH / fH, 0, dH - 1))

            patch = depth_map[
                max(0, dy - 2):min(dH, dy + 3),
                max(0, dx - 2):min(dW, dx + 3),
            ]
            if patch.size == 0:
                continue

            d_rel = float(np.median(patch))
            if d_rel < 1e-6:
                continue

            scales.append(Z_metric / d_rel)

        return scales


def load_depth_model(ckpt_path: str, encoder: str, device: str):
    cfg = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device).eval()
    print(f"[Depth] Loaded {encoder} checkpoint: {ckpt_path} (device={device})")
    return model


def depth_colormap(depth_map: np.ndarray) -> np.ndarray:
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min < 1e-6:
        norm = np.zeros_like(depth_map, dtype=np.uint8)
    else:
        norm = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    norm = 255 - norm
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)


def depth_to_color(depth_val: float, d_min: float, d_max: float):
    if d_max - d_min < 1e-6:
        t = 0.5
    else:
        t = 1.0 - (depth_val - d_min) / (d_max - d_min)
    b = int((1 - t) * 255)
    r = int(t * 255)
    return (b, 0, r)


def sample_depth_at_pixel(depth_map: np.ndarray, px: int, py: int, frame_shape: tuple) -> float:
    dH, dW = depth_map.shape
    fH, fW = frame_shape[:2]
    dx = int(np.clip(px * dW / fW, 0, dW - 1))
    dy = int(np.clip(py * dH / fH, 0, dH - 1))
    return float(depth_map[dy, dx])


def draw_joints_with_depth(
    frame: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    depth_map: np.ndarray,
    conf_thr: float = 0.3,
    show_name: bool = False,
    use_metric: bool = False,
):
    fh, fw = frame.shape[:2]
    d_min, d_max = float(depth_map.min()), float(depth_map.max())

    for idx, (xy, conf) in enumerate(zip(keypoints_xy, keypoints_conf)):
        if conf < conf_thr:
            continue

        px = int(np.clip(xy[0], 0, fw - 1))
        py = int(np.clip(xy[1], 0, fh - 1))

        d_val = sample_depth_at_pixel(depth_map, px, py, frame.shape)
        color = depth_to_color(d_val, d_min, d_max)

        cv2.circle(frame, (px, py), 6, color, -1)
        cv2.circle(frame, (px, py), 6, (255, 255, 255), 1)

        if use_metric:
            label = f"{d_val:.2f}m"
        else:
            label = f"{d_val:.2f}"

        if show_name:
            label = f"{KEYPOINT_NAMES[idx]}:{label}"

        cv2.putText(
            frame,
            label,
            (px + 8, py - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            4,
            cv2.LINE_AA,
        )


def draw_person_metric_boxes(frame, boxes_xyxy, boxes_conf, metric_depth_map, min_conf=0.5):
    if metric_depth_map is None:
        return

    fH, fW = frame.shape[:2]

    for box, conf in zip(boxes_xyxy, boxes_conf):
        if conf < min_conf:
            continue

        x1, y1, x2, y2 = box.astype(int)
        x1 = int(np.clip(x1, 0, fW - 1))
        y1 = int(np.clip(y1, 0, fH - 1))
        x2 = int(np.clip(x2, 0, fW - 1))
        y2 = int(np.clip(y2, 0, fH - 1))

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        z_m = sample_depth_at_pixel(metric_depth_map, cx, cy, frame.shape)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(
            frame,
            f"person {z_m:.2f}m",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            4,
            cv2.LINE_AA,
        )


def open_camera(source: str):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {source}")
    return cap


def get_intrinsics_from_args(args) -> CameraIntrinsics:
    if args.intrinsics_yaml:
        return CameraIntrinsics.from_yaml(args.intrinsics_yaml)

    return CameraIntrinsics(
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
    )


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Pose + Depth-Anything-V2 with approximate metric depth scaling"
    )

    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)

    parser.add_argument("--depth-model", type=str, default="checkpoints/depth_anything_v2_vits.pth")
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--depth-every", type=int, default=1)

    parser.add_argument("--show-depth", action="store_true")
    parser.add_argument("--show-names", action="store_true")
    parser.add_argument("--kp-conf", type=float, default=0.3)

    # Intrinsics: either pass YAML or pass numbers directly
    parser.add_argument("--intrinsics-yaml", type=str, default=None)
    parser.add_argument("--fx", type=float, default=1320.594)
    parser.add_argument("--fy", type=float, default=1320.594)
    parser.add_argument("--cx", type=float, default=958.032)
    parser.add_argument("--cy", type=float, default=721.8218)

    # Metric depth scaling params
    parser.add_argument("--person-height", type=float, default=1.70)
    parser.add_argument("--ema-alpha", type=float, default=0.15)
    parser.add_argument("--min-bbox-height", type=int, default=40)
    parser.add_argument("--metric-box-conf", type=float, default=0.5)

    # Optional: force ultralytics predict device
    parser.add_argument("--yolo-device", type=str, default=None,
                        help='Examples: "0", "cpu", "cuda:0"')

    args = parser.parse_args()

    DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[Device] {DEVICE}")

    intrinsics = get_intrinsics_from_args(args)
    scaler = MetricDepthScaler(
        intrinsics=intrinsics,
        person_height_m=args.person_height,
        ema_alpha=args.ema_alpha,
        min_bbox_height=args.min_bbox_height,
        min_conf=args.metric_box_conf,
    )

    model_pose = YOLO(args.model)
    model_depth = load_depth_model(args.depth_model, args.encoder, DEVICE)

    cap = open_camera(args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    cv2.namedWindow("Pose + Metric Depth", cv2.WINDOW_NORMAL)
    if args.show_depth:
        cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)

    depth_map_rel = None
    depth_map_metric = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        frame_count += 1

        if frame_count % args.depth_every == 0:
            with torch.no_grad():
                depth_map_rel = model_depth.infer_image(frame, args.input_size)

        predict_kwargs = dict(source=frame, conf=args.conf, verbose=False)
        if args.yolo_device is not None:
            predict_kwargs["device"] = args.yolo_device

        results = model_pose.predict(**predict_kwargs)
        result = results[0]
        display = result.plot()

        boxes_xyxy = np.empty((0, 4), dtype=np.float32)
        boxes_conf = np.empty((0,), dtype=np.float32)

        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
            boxes_conf = result.boxes.conf.detach().cpu().numpy()

        # Update metric scale using detected person boxes + relative depth map
        if depth_map_rel is not None and len(boxes_xyxy) > 0:
            scaler.update(
                depth_map=depth_map_rel,
                boxes_xyxy=boxes_xyxy,
                boxes_conf=boxes_conf,
                frame_shape=frame.shape,
            )
            depth_map_metric = scaler.to_metric(depth_map_rel)
        elif depth_map_rel is not None and scaler.scale is not None:
            depth_map_metric = scaler.to_metric(depth_map_rel)

        # Draw metric bbox labels
        if len(boxes_xyxy) > 0:
            draw_person_metric_boxes(
                display,
                boxes_xyxy,
                boxes_conf,
                depth_map_metric,
                min_conf=args.metric_box_conf,
            )

        # Draw per-joint depth in meters if available, otherwise relative depth
        if result.keypoints is not None and depth_map_rel is not None:
            kp_data = result.keypoints
            xy_all = kp_data.xy.detach().cpu().numpy()
            conf_all = kp_data.conf.detach().cpu().numpy()

            joint_depth_map = depth_map_metric if depth_map_metric is not None else depth_map_rel
            use_metric = depth_map_metric is not None

            for person_xy, person_conf in zip(xy_all, conf_all):
                draw_joints_with_depth(
                    display,
                    person_xy,
                    person_conf,
                    joint_depth_map,
                    conf_thr=args.kp_conf,
                    show_name=args.show_names,
                    use_metric=use_metric,
                )

        # HUD
        cv2.putText(display, f"Source: {args.source}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
        cv2.putText(display, f"Depth encoder: {args.encoder}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 200), 4)
        cv2.putText(display, f"Metric scale: {scaler.scale:.5f} m/unit" if scaler.scale is not None else "Metric scale: estimating...",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)
        cv2.putText(display, "Press q to quit",
                    (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)

        cv2.imshow("Pose + Metric Depth", display)

        if args.show_depth and depth_map_rel is not None:
            depth_vis_source = depth_map_metric if depth_map_metric is not None else depth_map_rel
            depth_vis = depth_colormap(depth_vis_source)
            depth_vis = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]))
            cv2.imshow("Depth Map", depth_vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()