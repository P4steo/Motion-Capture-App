import cv2
import mediapipe as mp
from bone_detect import AI_BoneDetector


class MediaPipeDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

    def detect(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame

class DefaultDetector:
    def __init__(self):
        pass  # Możesz dodać logikę inicjalizacji

    def detect(self, frame):
        # Tu dodaj swoją obecną metodę (np. z GUI)
        # Tymczasowo tylko zwraca surową klatkę
        return frame

def start_capture(width=640, height=480, mapping_type='default'):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    # Wybór detektora
    if mapping_type == 'mediapipe':
        detector = MediaPipeDetector()
    elif mapping_type == 'AI_detect':
        detector = AI_BoneDetector()
    else:
        detector = DefaultDetector()

    cv2.namedWindow('Motion Capture', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = detector.detect(frame)

        cv2.imshow('Motion Capture', frame)
        cv2.resizeWindow('Motion Capture', width, height)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
