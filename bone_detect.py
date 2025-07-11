import cv2
import numpy as np

# OpenPose BODY_25 keypoint indices:
# 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist, 5: LShoulder, 6: LElbow, 7: LWrist
# 8: MidHip, 9: RHip, 10: RKnee, 11: RAnkle, 12: LHip, 13: LKnee, 14: LAnkle
# 15: REye, 16: LEye, 17: REar, 18: LEar, 19: LBigToe, 20: LSmallToe, 21: LHeel
# 22: RBigToe, 23: RSmallToe, 24: RHeel

OPENPOSE_BONE_MAP = {
    "root": 8,           # MidHip
    "spine_base": 8,     # MidHip
    "spine_mid": 1,      # Neck
    "neck": 1,           # Neck
    "head": 0,           # Nose
    "left_shoulder": 5,  # LShoulder
    "left_upperarm": 6,  # LElbow
    "left_forearm": 7,   # LWrist
    "right_shoulder": 2, # RShoulder
    "right_upperarm": 3, # RElbow
    "right_forearm": 4,  # RWrist
    "left_hip": 12,      # LHip
    "left_thigh": 13,    # LKnee
    "left_calf": 14,     # LAnkle
    "left_foot": 21,     # LHeel
    "right_hip": 9,      # RHip
    "right_thigh": 10,   # RKnee
    "right_calf": 11,    # RAnkle
    "right_foot": 24     # RHeel
}

OPENPOSE_CONNECTIONS = [
    ("root", "spine_mid"),
    ("spine_mid", "neck"),
    ("neck", "head"),
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_upperarm"),
    ("left_upperarm", "left_forearm"),
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_upperarm"),
    ("right_upperarm", "right_forearm"),
    ("root", "left_hip"),
    ("left_hip", "left_thigh"),
    ("left_thigh", "left_calf"),
    ("left_calf", "left_foot"),
    ("root", "right_hip"),
    ("right_hip", "right_thigh"),
    ("right_thigh", "right_calf"),
    ("right_calf", "right_foot"),
]

class OpenPoseBoneDetector:
    def __init__(self):
        pass

    def detect(self, frame, keypoints):
        """
        Draw skeleton bones on frame using OpenPose BODY_25 keypoints.
        :param frame: np.ndarray, OpenCV frame (BGR)
        :param keypoints: list of 25 elements [x, y, confidence] (pixel coordinates)
        :return: frame with bones drawn
        """
        bone_points = {}
        for name, idx in OPENPOSE_BONE_MAP.items():
            if idx >= len(keypoints):
                continue
            x, y, conf = keypoints[idx]
            if conf > 0.05:  # Only plot if confidence is high enough
                bone_points[name] = (int(x), int(y))
        # Draw bones
        for a, b in OPENPOSE_CONNECTIONS:
            if a in bone_points and b in bone_points:
                cv2.line(frame, bone_points[a], bone_points[b], (0, 255, 0), 2)
        # Optionally: draw joints
        for pt in bone_points.values():
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)
        return frame