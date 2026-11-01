from __future__ import annotations

from typing import Sequence, Tuple

SkeletonEdge = Tuple[int, int]

# COCO 17-keypoint skeleton edges.
# Indices follow the standard COCO order:
#   0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
#   5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
#   9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
#   13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle.
COCO_17_EDGES: Tuple[SkeletonEdge, ...] = (
    (0, 1),   # nose → left_eye
    (0, 2),   # nose → right_eye
    (1, 3),   # left_eye → left_ear
    (2, 4),   # right_eye → right_ear
    (5, 7),   # left_shoulder → left_elbow
    (7, 9),   # left_elbow → left_wrist
    (6, 8),   # right_shoulder → right_elbow
    (8, 10),  # right_elbow → right_wrist
    (5, 6),   # left_shoulder → right_shoulder
    (5, 11),  # left_shoulder → left_hip
    (6, 12),  # right_shoulder → right_hip
    (11, 13), # left_hip → left_knee
    (13, 15), # left_knee → left_ankle
    (12, 14), # right_hip → right_knee
    (14, 16), # right_knee → right_ankle
    (11, 12), # left_hip → right_hip
)

# AP-10K 17-keypoint skeleton edges.
# Indices follow the AP-10K order:
#   0 left_eye, 1 right_eye, 2 nose, 3 neck,
#   4 left_front_shoulder, 5 right_front_shoulder,
#   6 left_front_paw, 7 right_front_paw,
#   8 left_back_shoulder, 9 right_back_shoulder,
#   10 left_back_paw, 11 right_back_paw, 12 tail,
#   13 left_ear, 14 right_ear, 15 head_top, 16 mid_back.
AP10K_EDGES: Tuple[SkeletonEdge, ...] = (
    (0, 1),   # left_eye → right_eye
    (0, 2),   # left_eye → nose
    (1, 2),   # right_eye → nose
    (2, 3),   # nose → neck
    (3, 4),   # neck → left_front_shoulder
    (3, 5),   # neck → right_front_shoulder
    (4, 6),   # left_front_shoulder → left_front_paw
    (5, 7),   # right_front_shoulder → right_front_paw
    (3, 8),   # neck → left_back_shoulder
    (3, 9),   # neck → right_back_shoulder
    (8, 10),  # left_back_shoulder → left_back_paw
    (9, 11),  # right_back_shoulder → right_back_paw
    (8, 12),  # left_back_shoulder → tail
    (9, 12),  # right_back_shoulder → tail
    (2, 13),  # nose → left_ear
    (2, 14),  # nose → right_ear
    (2, 15),  # nose → head_top
    (4, 5),   # left_front_shoulder → right_front_shoulder
    (16, 8),  # mid_back → left_back_shoulder
    (16, 9),  # mid_back → right_back_shoulder
)
