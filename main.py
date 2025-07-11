import sys
import cv2
import numpy as np
import mediapipe as mp
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QPushButton,
    QComboBox, QGroupBox, QMessageBox, QTabWidget, QTextEdit, QHBoxLayout, QFileDialog
)
from PyQt5.QtCore import QTimer, Qt, QTime
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
        self.setMinimumSize(1000, 700)
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

        # Zegar
        self.clock_lbl = QLabel()
        self.clock_lbl.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.clock_lbl.setStyleSheet("font-size:16px; color:#2980b9; background:rgba(255,255,255,0.8); padding:3px; border-radius:5px;")
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        self.update_clock()

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

        layout = QVBoxLayout()

        cam_grp = QGroupBox("🎥 Kamera")
        cam_layout = QHBoxLayout()

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

        self.start_btn = QPushButton("🚩 Start")
        self.start_btn.clicked.connect(self.toggle_capture)

        self.quick5_btn = QPushButton("Nagrywaj 5s")
        self.quick5_btn.clicked.connect(lambda: self.quick_record(5))
        self.quick10_btn = QPushButton("Nagrywaj 10s")
        self.quick10_btn.clicked.connect(lambda: self.quick_record(10))

        cam_layout.addWidget(QLabel("🔍 Rozdzielczość:"))
        cam_layout.addWidget(self.res_combo)
        cam_layout.addWidget(QLabel("⌚ Opóźnienie (s):"))
        cam_layout.addWidget(self.delay_combo)
        cam_layout.addWidget(QLabel("🦴 Mapowanie:"))
        cam_layout.addWidget(self.mapping_combo)
        cam_layout.addWidget(QLabel("🔄 Rotacje:"))
        cam_layout.addWidget(self.rotation_combo)
        cam_layout.addWidget(self.start_btn)
        cam_layout.addWidget(self.quick5_btn)
        cam_layout.addWidget(self.quick10_btn)

        cam_grp.setLayout(cam_layout)
        layout.addWidget(cam_grp)

        # Zegar w rogu (po starcie)
        layout.addWidget(self.clock_lbl, alignment=Qt.AlignRight | Qt.AlignTop)

        self.img_lbl = QLabel()
        self.img_lbl.setFixedSize(self.width, self.height)
        layout.addWidget(self.img_lbl, alignment=Qt.AlignCenter)
        self.status_lbl = QLabel("Status: gotowy")
        layout.addWidget(self.status_lbl)

        self.preview_btn = QPushButton("Podgląd skeletonu")
        self.preview_btn.clicked.connect(self.start_preview)
        layout.addWidget(self.preview_btn)
        self.main_tab.setLayout(layout)

        el = QVBoxLayout()
        btn_json = QPushButton("Zapisz JSON")
        btn_json.clicked.connect(self.export_json)
        btn_bvh = QPushButton("Eksportuj BVH")
        btn_bvh.clicked.connect(self.export_bvh)
        el.addWidget(btn_json)
        el.addWidget(btn_bvh)
        self.export_tab.setLayout(el)

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

    def set_rotation_mode(self, idx):
        self.rotation_mode = '2D' if idx == 0 else '3D'

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
        self.timer.start(33)
        self.status_lbl.setText("Przechwytywanie")

    def quick_record(self, seconds):
        if self.capturing:
            return
        self.delay = 0
        self.record_seconds = seconds
        self.status_lbl.setText(f"Nagrywanie {seconds}s...")
        self.start_capture()
        self.record_timer = QTimer()
        self.record_timer.setSingleShot(True)
        self.record_timer.timeout.connect(self.stop_quick_record)
        self.record_timer.start(seconds * 1000)

    def stop_quick_record(self):
        self.toggle_capture()
        self.status_lbl.setText("Zakończono szybkie nagrywanie.")

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

    def update_clock(self):
        from datetime import datetime
        self.clock_lbl.setText(datetime.now().strftime("%H:%M:%S"))

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