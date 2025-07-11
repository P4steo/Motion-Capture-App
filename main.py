import sys
import cv2
import numpy as np
import mediapipe as mp
import json
import datetime
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QPushButton,
    QComboBox, QGroupBox, QMessageBox, QTabWidget, QTextEdit, QHBoxLayout
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon

# Map MediaPipe landmarks to Unreal UE5 bone names
UE5_BONE_MAP = {
    'pelvis': mp.solutions.pose.PoseLandmark.LEFT_HIP.value,
    'spine_01': mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
    'spine_02': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
    'spine_03': mp.solutions.pose.PoseLandmark.NOSE.value,
    'clavicle_l': mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
    'upperarm_l': mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value,
    'lowerarm_l': mp.solutions.pose.PoseLandmark.LEFT_WRIST.value,
    'hand_l': mp.solutions.pose.PoseLandmark.LEFT_INDEX.value,
    'clavicle_r': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
    'upperarm_r': mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value,
    'lowerarm_r': mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value,
    'hand_r': mp.solutions.pose.PoseLandmark.RIGHT_INDEX.value,
    'neck_01': mp.solutions.pose.PoseLandmark.NOSE.value,
    'head': mp.solutions.pose.PoseLandmark.NOSE.value,
    'thigh_l': mp.solutions.pose.PoseLandmark.LEFT_KNEE.value,
    'calf_l': mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value,
    'foot_l': mp.solutions.pose.PoseLandmark.LEFT_HEEL.value,
    'ball_l': mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value,
    'thigh_r': mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value,
    'calf_r': mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value,
    'foot_r': mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value,
    'ball_r': mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value
}

HIERARCHY = [
    ('root', None),
    ('pelvis', 'root'),
    ('spine_01', 'pelvis'),
    ('spine_02', 'spine_01'),
    ('spine_03', 'spine_02'),
    ('clavicle_l', 'spine_03'),
    ('upperarm_l', 'clavicle_l'),
    ('lowerarm_l', 'upperarm_l'),
    ('hand_l', 'lowerarm_l'),
    ('clavicle_r', 'spine_03'),
    ('upperarm_r', 'clavicle_r'),
    ('lowerarm_r', 'upperarm_r'),
    ('hand_r', 'lowerarm_r'),
    ('neck_01', 'spine_03'),
    ('head', 'neck_01'),
    ('thigh_l', 'pelvis'),
    ('calf_l', 'thigh_l'),
    ('foot_l', 'calf_l'),
    ('ball_l', 'foot_l'),
    ('thigh_r', 'pelvis'),
    ('calf_r', 'thigh_r'),
    ('foot_r', 'calf_r'),
    ('ball_r', 'foot_r')
]


def calculate_angle(a, b, c):
    """Oblicz kąt ABC w stopniach."""
    ba = np.array([a['x'] - b['x'], a['y'] - b['y'], a['z'] - b['z']])
    bc = np.array([c['x'] - b['x'], c['y'] - b['y'], c['z'] - b['z']])
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


class MotionCaptureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Capture App")
        # Ustawienie ikony okna
        self.setWindowIcon(QIcon("E:\motion_capture_app\3208405.png"))

        self.setMinimumSize(1000, 700)

        self.cap = None
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

        self.width, self.height = 640, 480
        self.capturing = False
        self.landmark_data = []
        self.preview_index = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.preview_frame)

        self.init_ui()

    def init_ui(self):

        app.setStyleSheet("""
            QPushButton {
                background-color: #f9f9f9;
                color: #2c3e50;
                font-weight: 600;
                font-size: 14px;
                border: 1.5px solid #2980b9;
                border-radius: 8px;
                padding: 8px 20px;
                min-width: 100px;
                min-height: 35px;
                box-shadow: 0 3px 5px rgba(41, 128, 185, 0.3);
                transition: all 0.3s ease;
            }
            QPushButton:hover {
                background-color: #3498db;
                color: white;
                border-color: #2980b9;
                box-shadow: 0 4px 7px rgba(41, 128, 185, 0.5);
            }
            QPushButton:pressed {
                background-color: #2980b9;
                box-shadow: none;
            }
        """)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        self.main_tab, self.export_tab, self.help_tab = QWidget(), QWidget(), QWidget()
        tabs.addTab(self.main_tab, "🖥️ Główny")
        tabs.addTab(self.export_tab, "📤 Eksport")
        tabs.addTab(self.help_tab, "🛟 Pomoc")

        # Główny
        layout = QVBoxLayout()

        cam_grp = QGroupBox("🎥 Kamera")
        cam_layout = QHBoxLayout()

        self.res_combo = QComboBox()
        self.res_combo.addItems(["640x480", "1280x720", "1920x1080"])

        self.delay_combo = QComboBox()
        self.delay_combo.addItems(["0", "3", "5", "10"])

        self.mapping_combo = QComboBox()
        self.mapping_combo.addItems(["Mediapipe", "AI_Detect"])
        self.mapping_combo.currentTextChanged.connect(self.set_mapping_type)

        self.start_btn = QPushButton("🚩 Start")
        self.start_btn.clicked.connect(self.toggle_capture)

        cam_layout.addWidget(QLabel("🔍 Rozdzielczość:"))
        cam_layout.addWidget(self.res_combo)
        cam_layout.addWidget(QLabel("⌚ Opóźnienie (s):"))
        cam_layout.addWidget(self.delay_combo)
        cam_layout.addWidget(QLabel("🦴 Mapowanie:"))
        cam_layout.addWidget(self.mapping_combo)
        cam_layout.addWidget(self.start_btn)

        cam_grp.setLayout(cam_layout)
        layout.addWidget(cam_grp)

        self.img_lbl = QLabel()
        self.img_lbl.setFixedSize(self.width, self.height)
        layout.addWidget(self.img_lbl, alignment=Qt.AlignCenter)
        self.status_lbl = QLabel("Status: gotowy")
        layout.addWidget(self.status_lbl)

        self.preview_btn = QPushButton("Podgląd skeletonu")
        self.preview_btn.clicked.connect(self.start_preview)
        layout.addWidget(self.preview_btn)
        self.main_tab.setLayout(layout)

        # Eksport
        el = QVBoxLayout()
        btn_json = QPushButton("Zapisz JSON")
        btn_json.clicked.connect(self.export_json)
        btn_bvh = QPushButton("Eksportuj BVH")
        btn_bvh.clicked.connect(self.export_bvh)
        el.addWidget(btn_json)
        el.addWidget(btn_bvh)
        self.export_tab.setLayout(el)

        # Pomoc
        hl = QVBoxLayout()
        ht = QTextEdit()
        ht.setReadOnly(True)
        ht.setPlainText("Import BVH: File > Import > Motion Capture (.bvh)\n\n"
                        "Suggested workflow:\n"
                        "1. Nagrywaj ruch w głównym widoku.\n"
                        "2. Eksportuj BVH w zakładce Eksport.\n"
                        "3. Zaimportuj do Blendera dla animacji swojej postaci.")
        hl.addWidget(ht)
        self.help_tab.setLayout(hl)

    def set_mapping_type(self, mapowanie):
        print(f"Wybrano mapowanie: {mapowanie}")
        # dodać logikę zmiany mapowania tutaj

    def toggle_capture(self):
        if not self.capturing:
            self.delay = int(self.delay_combo.currentText())
            self.status_lbl.setText(f"Start za {self.delay}s")

            # Timer do odliczania delaya
            self.delay_timer = QTimer()
            self.delay_timer.timeout.connect(self.update_delay_countdown)
            self.delay_timer.start(1000)  # co 1 sekundę

        else:
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.start_btn.setText("Start")
            self.capturing = False
            self.status_lbl.setText("Zatrzymano")

    def update_delay_countdown(self):
        if self.delay > 1:
            self.delay -= 1
            self.status_lbl.setText(f"Start za {self.delay}s")
        else:
            self.delay_timer.stop()
            self.status_lbl.setText("Start!")
            self.start_capture()

    def start_capture(self):
        self.landmark_data.clear()
        self.capturing = True
        self.start_btn.setText("🛑 Stop")
        w, h = map(int, self.res_combo.currentText().split('x'))
        self.width, self.height = w, h
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.timer.start(33)
        self.status_lbl.setText("Przechwytywanie")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if res.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, res.pose_landmarks,
                                           self.mp_pose.POSE_CONNECTIONS)
            ld = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in res.pose_landmarks.landmark]
            self.landmark_data.append(ld)
            left_elbow = calculate_angle(
                ld[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                ld[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                ld[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            )
            left_knee = calculate_angle(
                ld[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                ld[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                ld[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            )
            print(f"Elbow angle: {left_elbow:.1f}°, Knee angle: {left_knee:.1f}°")
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        qt = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        self.img_lbl.setPixmap(QPixmap.fromImage(qt).scaled(self.img_lbl.size(), Qt.KeepAspectRatio))

    def start_preview(self):
        if not self.landmark_data:
            QMessageBox.warning(self, "Brak danych", "Najpierw nagraj dane.️")
            return
        self.preview_index = 0
        self.preview_timer.start(100)

    def preview_frame(self):
        if self.preview_index >= len(self.landmark_data):
            self.preview_timer.stop()
            return
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        landmarks = self.landmark_data[self.preview_index]
        for name, parent in HIERARCHY:
            if parent and name in UE5_BONE_MAP and parent in UE5_BONE_MAP:
                i0, i1 = UE5_BONE_MAP[name], UE5_BONE_MAP[parent]
                if i0 < len(landmarks) and i1 < len(landmarks):
                    x0, y0 = int(landmarks[i0]['x'] * self.width), int(landmarks[i0]['y'] * self.height)
                    x1, y1 = int(landmarks[i1]['x'] * self.width), int(landmarks[i1]['y'] * self.height)
                    cv2.line(frame, (x1, y1), (x0, y0), (0, 150, 0), 2)
        h, w, ch = frame.shape
        qt = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.img_lbl.setPixmap(QPixmap.fromImage(qt).scaled(self.img_lbl.size(), Qt.KeepAspectRatio))
        self.preview_index += 1

    def export_json(self):
        if not self.landmark_data:
            QMessageBox.warning(self, "⚠️Brak danych", "Najpierw nagraj dane.")
            return
        fn = f"mocap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fn, 'w') as f:
            json.dump(self.landmark_data, f, indent=2)
        QMessageBox.information(self, "✅ Zapisano", fn)

    def export_bvh(self):
        if not self.landmark_data:
            QMessageBox.warning(self, "⚠️Brak danych", "Najpierw nagraj dane.")
            return

        def write_bvh(filename, frames):
            with open(filename, 'w') as f:
                f.write("HIERARCHY\n")
                for bone, parent in HIERARCHY:
                    f.write(f"// {bone} parent: {parent}\n")
                f.write("MOTION\n")
                f.write(f"Frames: {len(frames)}\n")
                f.write("Frame Time: 0.0333333\n")
                for frame in frames:
                    pelvis = frame[UE5_BONE_MAP['pelvis']]
                    x, y, z = pelvis['x'] * 100, pelvis['y'] * 100, pelvis['z'] * 100
                    line = f"{x:.2f} {y:.2f} {z:.2f} 0.00 0.00 0.00"
                    for _ in HIERARCHY[1:]:
                        line += " 0.00 0.00 0.00"
                    f.write(line + "\n")

        fn = f"mocap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.bvh"
        write_bvh(fn, self.landmark_data)
        QMessageBox.information(self, "✅ Zapisano BVH", fn)

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QFont()
    font.setPointSize(11)
    app.setFont(font)
    window = MotionCaptureApp()
    window.show()
    sys.exit(app.exec_())
