# Real-Time Monocular 3D Human Pose and Metric Depth Reconstruction

This project builds a real-time monocular 3D reconstruction pipeline that estimates approximate metric depth and 3D human joint positions from a single RGB phone camera stream.

The system combines **YOLOv8-Pose**, **Depth Anything V2**, camera intrinsic geometry, and human-height-based scale recovery to lift 2D human pose observations into approximate 3D camera coordinates.

📄 **Full Project Report:** 

---

## Overview

Recovering 3D human motion from monocular video is challenging because a single RGB camera does not directly provide metric depth. Monocular depth models can predict relative depth, but the output is not immediately usable as real-world distance.

This project addresses this problem by using the detected human body as a scale reference. Given an RGB frame, the system detects 2D human keypoints, estimates a relative depth map, computes an approximate metric scale from the person’s image-space height, and reconstructs 3D joint positions using camera intrinsics.

---

## Pipeline

The proposed pipeline processes each frame in real time:

1. Capture an RGB frame from a phone camera.
2. Detect 2D human body keypoints using YOLOv8-Pose.
3. Estimate a dense relative depth map using Depth Anything V2.
4. Estimate the person’s image-space height from detected body keypoints.
5. Recover metric scale using the pinhole camera model and an assumed real-world human height.
6. Convert relative depth into approximate metric depth.
7. Back-project 2D keypoints into 3D camera coordinates.
8. Visualize the reconstructed 3D human pose and aligned depth map.

```text
Phone RGB Stream
        |
        v
+-------------------+
|  YOLOv8-Pose      |
|  2D Keypoints     |
+-------------------+
        |
        |                  
        v                  
+-------------------+      
| Depth Anything V2 |
| Relative Depth    |
+-------------------+
        |
        v
+-----------------------------+
| Human-Height Scale Recovery |
+-----------------------------+
        |
        v
+-----------------------------+
| 3D Back-Projection          |
+-----------------------------+
        |
        v
+-----------------------------+
| Real-Time Visualization     |
+-----------------------------+
