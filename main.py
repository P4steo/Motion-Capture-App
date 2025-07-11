import sys
import cv2
import numpy as np
import mediapipe as mp
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QPushButton,
    QComboBox, QGroupBox, QMessageBox, QTabWidget, QTextEdit, QHBoxLayout, QFileDialog, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

from bone_detect import OpenPoseBoneDetector

MEDIAPIPE_TO_OPENPOSE = [
    0, 0, 12, 14, 16, 11, 13, 15, 24, 23, 25, 27, 24, 26, 28, 2, 5,
    7, 8, 31, 32, 29, 28, 30, 27
]

def mediapipe_keypoints_to_openpose(mp_landmarks):
    openpose_points = []
    if len(mp_landmarks) < 25:
        return [[0, 0, 0, 0]] * 25
    if len(mp_landmarks) > 24:
        midhip_x = (mp_landmarks[23]['x'] + mp_landmarks[24]['x']) / 2
        midhip_y = (mp_landmarks[23]['y'] + mp_landmarks[24]['y']) / 2
        midhip_z = (mp_landmarks[23]['z'] + mp_landmarks[24]['z']) / 2
        midhip_vis = (mp_landmarks[23]['visibility'] + mp_landmarks[24]['visibility']) / 2
    else:
        midhip_x, midhip_y, midhip_z, midhip_vis = 0, 0, 0, 0
    for idx in range(25):
        mp_idx = MEDIAPIPE_TO_OPENPOSE[idx]
        if idx == 8:
            openpose_points.append([midhip_x, midhip_y, midhip_z, midhip_vis])
        elif mp_idx >= len(mp_landmarks):
            openpose_points.append([0, 0, 0, 0])
        else:
            lm = mp_landmarks[mp_idx]
            openpose_points.append([
                lm.get('x', 0), lm.get('y', 0), lm.get('z', 0), lm.get('visibility', 0)
            ])
    return openpose_points

def vector_angle(v1, v2):
    a = np.array(v1)
    b = np.array(v2)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cosang = np.clip(cosang, -1, 1)
    angle = np.arccos(cosang)
    return np.degrees(angle)

def get_simple_2d_rotations(points):
    rots = []
    rots.append([0.0, 0.0, 0.0])
    v_spine = [points[1][0] - points[8][0], points[1][1] - points[8][1]]
    rots.append([vector_angle([0, -1], v_spine), 0.0, 0.0])
    v_neckhead = [points[0][0] - points[1][0], points[0][1] - points[1][1]]
    rots.append([vector_angle([0, -1], v_neckhead), 0.0, 0.0])
    rots.append([0.0, 0.0, 0.0])
    v_ls_le = [points[6][0] - points[5][0], points[6][1] - points[5][1]]
    rots.append([vector_angle([1, 0], v_ls_le), 0.0, 0.0])
    v_le_lw = [points[7][0] - points[6][0], points[7][1] - points[6][1]]
    rots.append([vector_angle([1, 0], v_le_lw), 0.0, 0.0])
    rots.append([0.0, 0.0, 0.0])
    v_rs_re = [points[3][0] - points[2][0], points[3][1] - points[2][1]]
    rots.append([vector_angle([-1, 0], v_rs_re), 0.0, 0.0])
    v_re_rw = [points[4][0] - points[3][0], points[4][1] - points[3][1]]
    rots.append([vector_angle([-1, 0], v_re_rw), 0.0, 0.0])
    rots.append([0.0, 0.0, 0.0])
    v_lh_lk = [points[13][0] - points[12][0], points[13][1] - points[12][1]]
    rots.append([vector_angle([0, 1], v_lh_lk), 0.0, 0.0])
    v_lk_la = [points[14][0] - points[13][0], points[14][1] - points[13][1]]
    rots.append([vector_angle([0, 1], v_lk_la), 0.0, 0.0])
    rots.append([0.0, 0.0, 0.0])
    v_rh_rk = [points[10][0] - points[9][0], points[10][1] - points[9][1]]
    rots.append([vector_angle([0, 1], v_rh_rk), 0.0, 0.0])
    v_rk_ra = [points[11][0] - points[10][0], points[11][1] - points[10][1]]
    rots.append([vector_angle([0, 1], v_rk_ra), 0.0, 0.0])
    rots.append([0.0, 0.0, 0.0])
    return rots

def get_bvh_rotations(points):
    if points is None or len(points) < 25:
        return [[0, 0, 0]] * 16
    bones = [
        (8, 8, 1,  "Spine"),
        (1, 1, 0,  "Neck"),
        (5, 5, 6,  "LeftShoulder"),
        (6, 6, 7,  "LeftElbow"),
        (2, 2, 3,  "RightShoulder"),
        (3, 3, 4,  "RightElbow"),
        (12,12,13, "LeftHip"),
        (13,13,14, "LeftKnee"),
        (9, 9,10,  "RightHip"),
        (10,10,11, "RightKnee"),
    ]
    def rotation_from_three_points(parent, joint, child):
        v1 = np.array(points[joint][:3]) - np.array(points[parent][:3])
        v2 = np.array(points[child][:3]) - np.array(points[joint][:3])
        if np.linalg.norm(v1) < 1e-4 or np.linalg.norm(v2) < 1e-4:
            return [0.0, 0.0, 0.0]
        z_axis = v1 / np.linalg.norm(v1)
        x_axis = np.cross(z_axis, v2)
        if np.linalg.norm(x_axis) < 1e-4:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotmat = np.stack([x_axis, y_axis, z_axis], axis=1)
        sy = np.sqrt(rotmat[0, 0] ** 2 + rotmat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(rotmat[2, 1], rotmat[2, 2])
            y = np.arctan2(-rotmat[2, 0], sy)
            z = np.arctan2(rotmat[1, 0], rotmat[0, 0])
        else:
            x = np.arctan2(-rotmat[1, 2], rotmat[1, 1])
            y = np.arctan2(-rotmat[2, 0], sy)
            z = 0
        euler = np.degrees([z, x, y])
        euler = [0.0 if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v for v in euler]
        return euler
    rots = [[0.0, 0.0, 0.0]]
    for parent, joint, child, _ in bones:
        rots.append(rotation_from_three_points(parent, joint, child))
    while len(rots) < 16:
        rots.append([0.0, 0.0, 0.0])
    return rots

class MotionCaptureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Capture App")
        self.setMinimumSize(1150, 750)
        self.cap = None
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.width, self.height = 1280, 720
        self.capturing = False
        self.landmark_data = []
        self.preview_index = 0

        self.rotation_mode = '2D'
        self.mapping_type = "Mediapipe"
        self.openpose_detector = OpenPoseBoneDetector()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.preview_frame)
        self.quick_record_pending = False

        self.export_formats = [
            ("BVH", "Eksportuj BVH (Blender, Maya, Unity)"),
            ("JSON", "Eksportuj JSON (surowe dane)"),
            ("FBX", "Eksportuj FBX (animacja 3D)"),
        ]
        self.init_ui()

    def init_ui(self):
        app.setStyleSheet("""
            QComboBox, QComboBox QAbstractItemView {
                background-color: #f0f6fa;
                color: #2c3e50;
                font-size: 15px;
                border: 2px solid #2980b9;
                border-radius: 8px;
                padding: 5px 18px 5px 10px;
                min-width: 120px;
            }
            QComboBox QAbstractItemView {
                selection-background-color: #b1d1f5;
                selection-color: #2c3e50;
            }
            QPushButton {
                background-color: #1FA5FF;
                color: white;
                font-weight: 600;
                font-size: 18px;
                border: none;
                border-radius: 12px;
                padding: 12px 40px;
                margin: 10px 0;
            }
            QPushButton#startStop {
                background-color: #32cb7c;
                font-size: 19px;
                min-width: 190px;
                font-weight: bold;
            }
            QPushButton#startStop[recording="true"] {
                background-color: #ff6565;
            }
            QPushButton:hover {
                background-color: #1761a0;
            }
            QGroupBox {
                font-size: 16px;
                font-weight: 600;
                border: 1.5px solid #b1d1f5;
                border-radius: 10px;
                margin-top: 12px;
                padding: 7px;
            }
        """)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        self.main_tab, self.export_tab, self.help_tab = QWidget(), QWidget(), QWidget()
        tabs.addTab(self.main_tab, "🖥️ Główny")
        tabs.addTab(self.export_tab, "📤 Eksport")
        tabs.addTab(self.help_tab, "🛟 Pomoc")

        # --- GŁÓWNY ---
        layout = QVBoxLayout()
        cam_grp = QGroupBox("🎥 Kamera")
        cam_layout = QHBoxLayout()
        cam_layout.setSpacing(18)

        self.res_combo = QComboBox()
        self.res_combo.addItems(["1280x720", "640x480", "1920x1080"])
        self.res_combo.setCurrentIndex(0)
        self.delay_combo = QComboBox()
        self.delay_combo.addItems(["0", "3", "5", "10"])
        self.mapping_combo = QComboBox()
        self.mapping_combo.addItems(["Mediapipe"])
        self.mapping_combo.currentTextChanged.connect(self.set_mapping_type)
        self.rotation_combo = QComboBox()
        self.rotation_combo.addItems(["Rotacje uproszczone (2D)", "Rotacje 3D (Euler)"])
        self.rotation_combo.currentIndexChanged.connect(self.set_rotation_mode)
        self.quick_combo = QComboBox()
        self.quick_combo.addItems(["Brak", "5 sekund", "10 sekund", "20 sekund"])
        self.quick_combo.setCurrentIndex(0)

        # Start/Stop jeden przycisk
        self.start_stop_btn = QPushButton("▶ Start")
        self.start_stop_btn.setObjectName("startStop")
        self.start_stop_btn.setProperty("recording", False)
        self.start_stop_btn.clicked.connect(self.toggle_capture)

        cam_layout.addWidget(QLabel("📏 Rozdzielczość:"))
        cam_layout.addWidget(self.res_combo)
        cam_layout.addWidget(QLabel("⏳ Opóźnienie (s):"))
        cam_layout.addWidget(self.delay_combo)
        cam_layout.addWidget(QLabel("🔧 Mapowanie:"))
        cam_layout.addWidget(self.mapping_combo)
        cam_layout.addWidget(QLabel("🔁 Rotacje:"))
        cam_layout.addWidget(self.rotation_combo)
        cam_layout.addWidget(QLabel("⚡ Szybkie nagrywanie:"))
        cam_layout.addWidget(self.quick_combo)
        cam_layout.addSpacerItem(QSpacerItem(30, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        cam_layout.addWidget(self.start_stop_btn)

        cam_grp.setLayout(cam_layout)
        layout.addWidget(cam_grp)

        self.img_lbl = QLabel()
        self.img_lbl.setFixedSize(self.width, self.height)
        layout.addWidget(self.img_lbl, alignment=Qt.AlignCenter)
        self.status_lbl = QLabel("Status: gotowy")
        self.status_lbl.setStyleSheet("font-size:15px; color:#1761a0; padding:5px;")
        layout.addWidget(self.status_lbl)
        self.preview_btn = QPushButton("Podgląd skeletonu")
        self.preview_btn.clicked.connect(self.start_preview)
        layout.addWidget(self.preview_btn)
        self.main_tab.setLayout(layout)

        # --- EKSPORT ---
        export_layout = QVBoxLayout()
        export_layout.setSpacing(20)
        self.export_combo = QComboBox()
        for code, label in self.export_formats:
            self.export_combo.addItem(label, code)
        self.export_combo.setCurrentIndex(0)
        self.export_btn = QPushButton("Eksportuj")
        self.export_btn.setStyleSheet("font-size:20px; background:#1FA5FF; border-radius:12px; padding:20px 0;")
        self.export_btn.clicked.connect(self.export_selected)
        self.export_info_lbl = QLabel("Wybierz format, by wyeksportować ruch do wybranego narzędzia animacji.")
        self.export_info_lbl.setStyleSheet("font-size:15px; color:#1761a0; padding:5px;")
        export_layout.addWidget(QLabel("Wybierz format eksportu:"))
        export_layout.addWidget(self.export_combo)
        export_layout.addWidget(self.export_btn)
        export_layout.addWidget(self.export_info_lbl)
        export_layout.addStretch()
        self.export_tab.setLayout(export_layout)

        # --- POMOC ---
        hl = QVBoxLayout()
        ht = QTextEdit()
        ht.setReadOnly(True)
        ht.setPlainText("Import BVH: File > Import > Motion Capture (.bvh)\n\n"
                        "Sugestie:\n"
                        "1. Nagrywaj ruch w głównym widoku.\n"
                        "2. Eksportuj BVH lub FBX w zakładce Eksport.\n"
                        "3. Zaimportuj do Blendera, Unity lub Maya do animacji postaci.")
        hl.addWidget(ht)
        self.help_tab.setLayout(hl)

    def set_mapping_type(self, selected_type):
        self.mapping_type = selected_type

    def set_rotation_mode(self, idx):
        self.rotation_mode = '2D' if idx == 0 else '3D'

    def toggle_capture(self):
        if not self.capturing:
            self.capturing = True
            self.start_stop_btn.setText("⏹ Stop")
            self.start_stop_btn.setProperty("recording", True)
            self.start_stop_btn.style().unpolish(self.start_stop_btn)
            self.start_stop_btn.style().polish(self.start_stop_btn)
            self.delay = int(self.delay_combo.currentText())
            quick_idx = self.quick_combo.currentIndex()
            if quick_idx == 0:
                self.quick_record_pending = False
                self.status_lbl.setText(f"Start za {self.delay}s")
            else:
                self.quick_record_pending = True
                self.quick_record_seconds = [0, 5, 10, 20][quick_idx]
                self.status_lbl.setText(f"Start za {self.delay}s, nagrywanie {self.quick_record_seconds}s...")
            self.delay_timer = QTimer()
            self.delay_timer.timeout.connect(self.update_delay_countdown)
            self.delay_timer.start(1000)
        else:
            self.stop_recording()

    def update_delay_countdown(self):
        if self.delay > 1:
            self.delay -= 1
            self.status_lbl.setText(f"Start za {self.delay}s" + (
                f", nagrywanie {self.quick_record_seconds}s..." if self.quick_record_pending else ""))
        else:
            self.delay_timer.stop()
            self.status_lbl.setText("Start!")
            self.start_capture()

    def start_capture(self):
        self.landmark_data.clear()
        w, h = map(int, self.res_combo.currentText().split('x'))
        self.width, self.height = w, h
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.timer.start(33)
        self.status_lbl.setText("Przechwytywanie")

        if self.quick_record_pending:
            self.record_timer = QTimer()
            self.record_timer.setSingleShot(True)
            self.record_timer.timeout.connect(self.stop_recording)
            self.record_timer.start(self.quick_record_seconds * 1000)
            self.quick_record_pending = False

    def stop_recording(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.capturing = False
        self.start_stop_btn.setText("▶ Start")
        self.start_stop_btn.setProperty("recording", False)
        self.start_stop_btn.style().unpolish(self.start_stop_btn)
        self.start_stop_btn.style().polish(self.start_stop_btn)
        self.status_lbl.setText("Nagrywanie zakończone.")

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)
            if res and res.pose_landmarks:
                mp_landmarks = []
                for lm in res.pose_landmarks.landmark:
                    mp_landmarks.append({
                        'x': lm.x * self.width,
                        'y': lm.y * self.height,
                        'z': lm.z * self.width,
                        'visibility': lm.visibility
                    })
                openpose_points = mediapipe_keypoints_to_openpose(mp_landmarks)
                frame = self.openpose_detector.detect(frame, openpose_points)
                self.landmark_data.append(mp_landmarks)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            qt = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
            self.img_lbl.setPixmap(QPixmap.fromImage(qt).scaled(self.img_lbl.size(), Qt.KeepAspectRatio))
        except Exception as e:
            print("Błąd w update_frame:", e)
            self.status_lbl.setText(f"Błąd: {e}")
            self.timer.stop()
            self.stop_recording()

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

    def export_selected(self):
        code = self.export_combo.currentData()
        if code == "BVH":
            self.export_bvh()
        elif code == "JSON":
            self.export_json()
        elif code == "FBX":
            self.export_fbx()
        else:
            QMessageBox.warning(self, "Eksport", "Nieznany format eksportu!")

    def export_json(self):
        if not self.landmark_data:
            QMessageBox.warning(self, "Brak danych", "Najpierw nagraj dane.")
            return
        options = QFileDialog.Options()
        fn, _ = QFileDialog.getSaveFileName(
            self, "Zapisz jako JSON", "mocap.json", "Pliki JSON (*.json)", options=options
        )
        if fn:
            with open(fn, 'w') as f:
                json.dump(self.landmark_data, f, indent=2)
            QMessageBox.information(self, "Zapisano", fn)

    def export_bvh(self):
        if not self.landmark_data:
            QMessageBox.warning(self, "Brak danych", "Najpierw nagraj dane.")
            return
        options = QFileDialog.Options()
        fn, _ = QFileDialog.getSaveFileName(
            self, "Zapisz jako BVH", "mocap.bvh", "Pliki BVH (*.bvh)", options=options
        )
        if fn:
            try:
                with open(fn, 'w') as f:
                    f.write("""HIERARCHY
ROOT Hips
{
    OFFSET 0.00 0.00 0.00
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {
        OFFSET 0.0 10.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Neck
        {
            OFFSET 0.0 10.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT Head
            {
                OFFSET 0.0 10.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {
                    OFFSET 0.0 7.0 0.0
                }
            }
        }
        JOINT LeftShoulder
        {
            OFFSET -5.0 10.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftElbow
            {
                OFFSET -15.0 0.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT LeftWrist
                {
                    OFFSET -20.0 0.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site
                    {
                        OFFSET -5.0 0.0 0.0
                    }
                }
            }
        }
        JOINT RightShoulder
        {
            OFFSET 5.0 10.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightElbow
            {
                OFFSET 15.0 0.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT RightWrist
                {
                    OFFSET 20.0 0.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site
                    {
                        OFFSET 5.0 0.0 0.0
                    }
                }
            }
        }
    }
    JOINT LeftHip
    {
        OFFSET -5.0 -10.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftKnee
        {
            OFFSET 0.0 -15.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftAnkle
            {
                OFFSET 0.0 -15.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {
                    OFFSET 0.0 -5.0 0.0
                }
            }
        }
    }
    JOINT RightHip
    {
        OFFSET 5.0 -10.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightKnee
        {
            OFFSET 0.0 -15.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightAnkle
            {
                OFFSET 0.0 -15.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {
                    OFFSET 0.0 -5.0 0.0
                }
            }
        }
    }
}
""")
                    f.write(f"MOTION\nFrames: {len(self.landmark_data)}\nFrame Time: 0.0333333\n")
                    base = self.landmark_data[0]
                    base_hips = [
                        (base[23]['x'] + base[24]['x']) / 2,
                        (base[23]['y'] + base[24]['y']) / 2,
                        (base[23]['z'] + base[24]['z']) / 2
                    ]
                    scale = 0.5
                    for frame in self.landmark_data:
                        points = mediapipe_keypoints_to_openpose(frame)
                        hips = [
                            (frame[23]['x'] + frame[24]['x']) / 2,
                            (frame[23]['y'] + frame[24]['y']) / 2,
                            (frame[23]['z'] + frame[24]['z']) / 2
                        ]
                        x = (hips[0] - base_hips[0]) * scale
                        y = (hips[1] - base_hips[1]) * scale
                        z = (hips[2] - base_hips[2]) * scale
                        out = [f"{x:.2f}", f"{y:.2f}", f"{z:.2f}"]
                        if self.rotation_mode == '2D':
                            rots = get_simple_2d_rotations(points)
                        else:
                            rots = get_bvh_rotations(points)
                        for rot in rots:
                            rot = [0.0 if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v for v in rot]
                            out += [f"{v:.2f}" for v in rot]
                        f.write(" ".join(out) + "\n")
                QMessageBox.information(self, "Zapisano BVH", fn)
            except Exception as e:
                QMessageBox.critical(self, "Błąd eksportu BVH", str(e))

    def export_fbx(self):
        QMessageBox.information(self, "Eksport FBX", "Eksport do FBX wymaga użycia zewnętrznych narzędzi (np. Blender, pyfbx). Możesz załadować BVH do Blendera i wyeksportować jako FBX.")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QFont()
    font.setPointSize(12)
    app.setFont(font)
    window = MotionCaptureApp()
    window.show()
    sys.exit(app.exec_())