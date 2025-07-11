# bone_detect.py
import cv2
import numpy as np

class AI_BoneDetector:
    def __init__(self):
        # Możesz tu podpiąć bardziej zaawansowany model AI, np. OpenPose
        # Tymczasowo użyjemy detekcji sylwetki (kontur + centroidy) jako uproszczony model AI
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100)

    def detect(self, frame):
        # Konwertujemy na szaro i wykrywamy kontury postaci
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor.apply(gray)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = frame.shape[:2]

        for cnt in contours:
            if cv2.contourArea(cnt) < 1000:
                continue

            # Prostokąt ograniczający kontur
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Szkielet heurystyczny: prosty model złożony z punktów ciała
            points = {
                "head": (x + w // 2, y + int(h * 0.1)),
                "shoulders": (x + w // 2, y + int(h * 0.3)),
                "torso": (x + w // 2, y + int(h * 0.5)),
                "hips": (x + w // 2, y + int(h * 0.7)),
                "left_leg": (x + int(w * 0.3), y + h),
                "right_leg": (x + int(w * 0.7), y + h)
            }

            # Rysuj linie między punktami
            self.draw_bones(frame, points)

        return frame

    def draw_bones(self, frame, pts):
        # Linie między sztucznie wyznaczonymi punktami
        pairs = [
            ("head", "shoulders"),
            ("shoulders", "torso"),
            ("torso", "hips"),
            ("hips", "left_leg"),
            ("hips", "right_leg")
        ]
        for a, b in pairs:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (255, 0, 0), 2)
                cv2.circle(frame, pts[a], 5, (0, 0, 255), -1)
                cv2.circle(frame, pts[b], 5, (0, 0, 255), -1)
