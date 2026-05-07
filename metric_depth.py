# ── New imports (top of file) ─────────────────────────────────────────────
import yaml
from metric_depth import CameraIntrinsics, MetricDepthScaler

# ── New CLI args (inside main, add to argparse) ───────────────────────────
parser.add_argument("--camera-cfg",    type=str, default="camera.yaml",
                    help="YAML file with camera intrinsics")
parser.add_argument("--person-height", type=float, default=1.70,
                    help="Known person height in metres")
parser.add_argument("--show-metric",   action="store_true",
                    help="Overlay metric depth (metres) instead of relative")

# ── After loading models (in main) ───────────────────────────────────────
with open(args.camera_cfg) as f:
    cam_cfg = yaml.safe_load(f)

intrinsics = CameraIntrinsics.from_yaml(args.camera_cfg)
scaler = MetricDepthScaler(
    intrinsics=intrinsics,
    person_height_m=cam_cfg.get("person_height_m", args.person_height),
    ema_alpha=cam_cfg.get("ema_alpha", 0.15),
)

# ── Inside the frame loop, after YOLO inference ──────────────────────────
# (replace the existing "Per-joint depth overlay" block)

if depth_map is not None and result.boxes is not None:
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    boxes_conf = result.boxes.conf.cpu().numpy()

    # Update metric scale estimate
    scale = scaler.update(depth_map, boxes_xyxy, boxes_conf, frame.shape)
    metric_map = scaler.to_metric(depth_map)  # None until first estimate

    # Choose which depth map to display / annotate
    active_map = metric_map if (args.show_metric and metric_map is not None) else depth_map

    if result.keypoints is not None and active_map is not None:
        xy_all   = result.keypoints.xy.cpu().numpy()
        conf_all = result.keypoints.conf.cpu().numpy()

        for person_xy, person_conf in zip(xy_all, conf_all):
            draw_joints_with_depth(
                display, person_xy, person_conf,
                active_map,
                conf_thr=args.kp_conf,
                show_name=args.show_names,
            )

# ── HUD: add scale display ────────────────────────────────────────────────
if scaler.scale is not None:
    cv2.putText(display, f"Scale: {scaler.scale:.3f} m/unit",
                (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 255), 1)