import sys
import cv2
import numpy as np
import mediapipe as mp
import json
import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QPushButton,
    QComboBox, QGroupBox, QMessageBox, QTabWidget, QTextEdit, QHBoxLayout
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

from bone_detect import OpenPoseBoneDetector  # Import rysowania szkieletu

# Mapowanie punktów MediaPipe (33) na OpenPose BODY_25 (przybliżenie)
MEDIAPIPE_TO_OPENPOSE = [
    0,   # Nose
    0,   # Neck (brak w MP - przyjmujemy Nose)
    12,  # RShoulder
    14,  # RElbow
    16,  # RWrist
    11,  # LShoulder
    13,  # LElbow
    15,  # LWrist
    24,  # MidHip (liczony osobno!)
    23,  # RHip
    25,  # RKnee
    27,  # RAnkle
    24,  # LHip
    26,  # LKnee
    28,  # LAnkle
    2,   # REye
    5,   # LEye
    7,   # REar
    8,   # LEar
    31,  # LBigToe
    32,  # LSmallToe
    29,  # LHeel
    28,  # RBigToe (przybliżenie)
    30,  # RSmallToe (przybliżenie)
    27,  # RHeel (przybliżenie)
]

def mediapipe_keypoints_to_openpose(mp_landmarks):
    """Konwertuje punkty MediaPipe na format OpenPose BODY_25 (przybliżenie). Wynik: lista 25 [x, y, confidence]"""
    openpose_points = []
    # MidHip = środek pomiędzy lewym i prawym biodrem
    if len(mp_landmarks) > 24:
        midhip_x = int((mp_landmarks[23]['x'] + mp_landmarks[24]['x']) / 2)
        midhip_y = int((mp_landmarks[23]['y'] + mp_landmarks[24]['y']) / 2)
        midhip_vis = (mp_landmarks[23]['visibility'] + mp_landmarks[24]['visibility']) / 2
    else:
        midhip_x, midhip_y, midhip_vis = 0, 0, 0
    for idx in range(25):
        mp_idx = MEDIAPIPE_TO_OPENPOSE[idx]
        if idx == 8:  # MidHip
            openpose_points.append([midhip_x, midhip_y, midhip_vis])
        elif mp_idx >= len(mp_landmarks):
            openpose_points.append([0, 0, 0])
        else:
            lm = mp_landmarks[mp_idx]
            openpose_points.append([lm['x'], lm['y'], lm['visibility']])
    return openpose_points

class MotionCaptureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Capture App")
        self.setMinimumSize(1000, 700)
        self.cap = None
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # NAJDOKŁADNIEJSZY model
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.width, self.height = 1280, 720  # Wyższa rozdzielczość domyślnie
        self.capturing = False
        self.landmark_data = []
        self.preview_index = 0

        self.mapping_type = "Mediapipe"
        self.openpose_detector = OpenPoseBoneDetector()

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
            }
            QPushButton:hover {
                background-color: #3498db;
                color: white;
                border-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #2980b9;
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
        self.res_combo.addItems(["1280x720", "640x480", "1920x1080"])
        self.res_combo.setCurrentIndex(0)  # Domyślnie 1280x720

        self.delay_combo = QComboBox()
        self.delay_combo.addItems(["0", "3", "5", "10"])

        self.mapping_combo = QComboBox()
        self.mapping_combo.addItems(["Mediapipe"])
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
                        "Sugestie:\n"
                        "1. Nagrywaj ruch w głównym widoku.\n"
                        "2. Eksportuj BVH w zakładce Eksport.\n"
                        "3. Zaimportuj do Blendera do animacji postaci.")
        hl.addWidget(ht)
        self.help_tab.setLayout(hl)

    def set_mapping_type(self, selected_type):
        self.mapping_type = selected_type
        print(f"Wybrano mapowanie: {selected_type}")

    def toggle_capture(self):
        if not self.capturing:
            self.delay = int(self.delay_combo.currentText())
            self.status_lbl.setText(f"Start za {self.delay}s")
            self.delay_timer = QTimer()
            self.delay_timer.timeout.connect(self.update_delay_countdown)
            self.delay_timer.start(1000)
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
        self.timer.start(33)  # 30 FPS
        self.status_lbl.setText("Przechwytywanie")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if res.pose_landmarks:
            # Pobierz punkty w px
            mp_landmarks = []
            for lm in res.pose_landmarks.landmark:
                mp_landmarks.append({
                    'x': int(lm.x * self.width),
                    'y': int(lm.y * self.height),
                    'z': lm.z,
                    'visibility': lm.visibility
                })
            # Mapuj na OpenPose-like
            openpose_points = mediapipe_keypoints_to_openpose(mp_landmarks)
            # Rysuj szkielet OpenPose
            frame = self.openpose_detector.detect(frame, openpose_points)
            self.landmark_data.append(mp_landmarks)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        qt = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        self.img_lbl.setPixmap(QPixmap.fromImage(qt).scaled(self.img_lbl.size(), Qt.KeepAspectRatio))

    def start_preview(self):
        if not self.landmark_data:
            QMessageBox.warning(self, "Brak danych", "Najpierw nagraj dane.")
            return
        self.preview_index = 0
        self.preview_timer.start(100)

    def preview_frame(self):
        if self.preview_index >= len(self.landmark_data):
            self.preview_timer.stop()
            return
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        mp_landmarks = self.landmark_data[self.preview_index]
        openpose_points = mediapipe_keypoints_to_openpose(mp_landmarks)
        frame = self.openpose_detector.detect(frame, openpose_points)
        h, w, ch = frame.shape
        qt = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.img_lbl.setPixmap(QPixmap.fromImage(qt).scaled(self.img_lbl.size(), Qt.KeepAspectRatio))
        self.preview_index += 1

    def export_json(self):
        if not self.landmark_data:
            QMessageBox.warning(self, "Brak danych", "Najpierw nagraj dane.")
            return
        fn = f"mocap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fn, 'w') as f:
            json.dump(self.landmark_data, f, indent=2)
        QMessageBox.information(self, "Zapisano", fn)

    def export_bvh(self):
        if not self.landmark_data:
            QMessageBox.warning(self, "Brak danych", "Najpierw nagraj dane.")
            return
        def write_bvh(filename, frames):
            with open(filename, 'w') as f:
                f.write("HIERARCHY\n// ...\nMOTION\n")
                f.write(f"Frames: {len(frames)}\n")
                f.write("Frame Time: 0.0333333\n")
                for frame in frames:
                    pelvis = frame[24]  # LHip (przybliżenie)
                    x, y, z = pelvis['x'], pelvis['y'], pelvis['z']
                    line = f"{x:.2f} {y:.2f} {z:.2f} 0.00 0.00 0.00"
                    for _ in range(24):
                        line += " 0.00 0.00 0.00"
                    f.write(line + "\n")
        fn = f"mocap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.bvh"
        write_bvh(fn, self.landmark_data)
        QMessageBox.information(self, "Zapisano BVH", fn)

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