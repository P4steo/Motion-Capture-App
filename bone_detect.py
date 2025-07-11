import cv2

# Mapowanie nazw "kości" OpenPose BODY_25 na indeksy keypointów
OPENPOSE_BONE_MAP = {
    "root": 8,           # MidHip
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

# Połączenia kości do rysowania (każda para to linia)
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
        Rysuje szkielet na obrazie na podstawie punktów BODY_25 w formacie [[x, y, conf], ...]
        :param frame: obraz OpenCV (BGR)
        :param keypoints: lista 25 punktów [x, y, confidence]
        :return: obraz z narysowanym szkieletem
        """
        bone_points = {}
        for name, idx in OPENPOSE_BONE_MAP.items():
            if idx >= len(keypoints):
                continue
            x, y, conf = keypoints[idx]
            if conf > 0.05:  # Rysuj tylko pewne punkty
                bone_points[name] = (int(x), int(y))
        # Rysuj połączenia
        for a, b in OPENPOSE_CONNECTIONS:
            if a in bone_points and b in bone_points:
                cv2.line(frame, bone_points[a], bone_points[b], (0, 255, 0), 2)
        # Rysuj punkty
        for pt in bone_points.values():
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)
        return frame