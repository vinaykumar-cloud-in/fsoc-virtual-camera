import streamlit as st
from collections import deque
from datetime import datetime, timezone
import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
from streamlit_image_coordinates import streamlit_image_coordinates
from src.detection.laser_spot_detector import LaserSpotDetector

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FSOC Virtual Camera Tracking Ground Station",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 360px !important;
            min-width: 360px !important;
        }
        section[data-testid="stSidebar"] > div {
            width: 360px !important;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            max-width: 100% !important;
        }
        section[data-testid="stSidebar"] button,
        section[data-testid="stSidebar"] [data-baseweb="select"] {
            width: 100% !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] * {
            white-space: normal !important;
            overflow-wrap: anywhere !important;
        }
        [role="listbox"] {
            min-width: 320px !important;
        }
        [role="option"] {
            white-space: normal !important;
            overflow-wrap: anywhere !important;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            word-break: normal !important;
        }
        [data-testid="stMetric"] {
            min-height: 76px;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            white-space: normal !important;
            overflow-wrap: anywhere !important;
        }
        h1, h2, h3 {
            overflow-wrap: anywhere;
        }
        img {
            max-width: 100%;
            height: auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


class KalmanFilter2D:
    def __init__(self, x0=320.0, y0=240.0):
        self.x = np.array([x0, y0, 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 25.0
        self.F = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        self.Q = np.eye(4) * 0.6
        self.R = np.eye(2) * 4.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, measurement):
        z = np.asarray(measurement, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].copy()

    def position(self):
        return self.x[:2].copy()

    def reset(self, x0=320.0, y0=240.0):
        self.x = np.array([x0, y0, 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 25.0


def _init_session_state():
    if "session_running" not in st.session_state:
        st.session_state.session_running = True
    if "telemetry_log" not in st.session_state:
        st.session_state.telemetry_log = []
    if "manual_target_x" not in st.session_state:
        st.session_state.manual_target_x = 320
    if "manual_target_y" not in st.session_state:
        st.session_state.manual_target_y = 240
    if "time_step" not in st.session_state:
        st.session_state.time_step = 0
    if "error_history" not in st.session_state:
        st.session_state.error_history = []
    if "time_history" not in st.session_state:
        st.session_state.time_history = []
    if "tracking_mode" not in st.session_state:
        st.session_state.tracking_mode = "Acquisition"
    if "roi_center" not in st.session_state:
        st.session_state.roi_center = None
    if "latest_frame_data" not in st.session_state:
        st.session_state.latest_frame_data = None
    if "velocity" not in st.session_state:
        st.session_state.velocity = (0.0, 0.0)
    if "last_known_pos" not in st.session_state:
        st.session_state.last_known_pos = (320.0, 240.0)
    if "camera_mode" not in st.session_state:
        st.session_state.camera_mode = "SIMULATION MODE"
    if "fov_deg" not in st.session_state:
        st.session_state.fov_deg = 20.0
    if "kalman" not in st.session_state:
        st.session_state.kalman = KalmanFilter2D()
    if "camera_cap" not in st.session_state:
        st.session_state.camera_cap = None


_init_session_state()

st.title("🛰️ FSOC Virtual Camera Tracking Ground Station")
st.caption("Real-time optical tracking console for synthetic camera, AI detection, and true 2D EKF filtering")

st.markdown(
    """
    <style>
        .status-pill {
            display: inline-block;
            padding: 0.45rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin: 0.35rem 0 1rem 0;
            border: 1px solid rgba(255,255,255,0.15);
        }
        .status-lock { background: rgba(0, 200, 120, 0.18); color: #8af0bf; }
        .status-loss { background: rgba(255, 100, 100, 0.14); color: #ff9a9a; }
        .status-predictive { background: rgba(255, 185, 80, 0.17); color: #ffd57f; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Simulation Controls")

camera_mode = st.sidebar.radio(
    "Camera Mode",
    ["SIMULATION MODE", "REAL CAMERA MODE"],
    index=0,
    help="Run the synthetic beacon or point a webcam at a physical laser spot on a white screen.",
)

turbulence = st.sidebar.slider(
    "Atmospheric Turbulence (Cn²)",
    0,
    10,
    2,
    help="Adds blur and shimmer to simulated frame",
)
target_speed = st.sidebar.slider(
    "Target Angular Speed",
    1,
    20,
    5,
    help="Speed of beacon movement",
)
motion_mode = st.sidebar.selectbox(
    "Target Motion Mode",
    ["Automatic (Circular)", "Manual Control"],
    help="Choose whether the beacon moves automatically or via manual inputs",
)

if motion_mode == "Manual Control":
    st.sidebar.caption("Click or drag in the target field, or steer with the D-pad below.")
    steering_step = st.sidebar.slider(
        "Steering Step (px per press)",
        1,
        40,
        10,
        help="How far each D-pad press moves the beacon",
    )

    def _nudge_target(dx, dy):
        st.session_state.manual_target_x = int(
            np.clip(st.session_state.manual_target_x + dx, 50, 590)
        )
        st.session_state.manual_target_y = int(
            np.clip(st.session_state.manual_target_y + dy, 50, 430)
        )

    dpad_top = st.sidebar.columns([1, 1, 1])
    with dpad_top[1]:
        if st.button("⬆️", key="steer_up", width="stretch"):
            _nudge_target(0, -steering_step)
    dpad_mid = st.sidebar.columns([1, 1, 1])
    with dpad_mid[0]:
        if st.button("⬅️", key="steer_left", width="stretch"):
            _nudge_target(-steering_step, 0)
    with dpad_mid[1]:
        if st.button("⏹️", key="steer_center", width="stretch"):
            st.session_state.manual_target_x = 320
            st.session_state.manual_target_y = 240
    with dpad_mid[2]:
        if st.button("➡️", key="steer_right", width="stretch"):
            _nudge_target(steering_step, 0)
    dpad_bottom = st.sidebar.columns([1, 1, 1])
    with dpad_bottom[1]:
        if st.button("⬇️", key="steer_down", width="stretch"):
            _nudge_target(0, steering_step)

    picker_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(picker_frame, (290, 240), (350, 240), (0, 180, 180), 1)
    cv2.line(picker_frame, (320, 210), (320, 270), (0, 180, 180), 1)
    cv2.circle(
        picker_frame,
        (st.session_state.manual_target_x, st.session_state.manual_target_y),
        12,
        (0, 255, 255),
        2,
    )
    selected_point = streamlit_image_coordinates(
        picker_frame,
        width=320,
        height=240,
        key="manual_target_picker",
        click_and_drag=True,
        cursor="crosshair",
    )
    point_x = None
    point_y = None
    if selected_point:
        point_x = selected_point.get("x", selected_point.get("x2"))
        point_y = selected_point.get("y", selected_point.get("y2"))
        if point_x is None or point_y is None:
            point_x = selected_point.get("x1")
            point_y = selected_point.get("y1")
        if point_x is None or point_y is None:
            point_x = point_y = None

    if point_x is not None and point_y is not None:
        displayed_width = selected_point.get("width", 640)
        displayed_height = selected_point.get("height", 480)
        st.session_state.manual_target_x = int(
            np.clip(point_x * 640 / displayed_width, 50, 590)
        )
        st.session_state.manual_target_y = int(
            np.clip(point_y * 480 / displayed_height, 50, 430)
        )

manual_target_x = st.session_state.manual_target_x
manual_target_y = st.session_state.manual_target_y

detector = st.sidebar.selectbox(
    "Detection Algorithm",
    ["YOLOv8 Beacon Simulation", "Bright-Spot Threshold", "Detector Disabled"],
    help="Select the detection mode for the laser spot tracker on the white-plane camera feed.",
)
receiver_shake = st.sidebar.slider(
    "Receiver Platform Shake",
    0,
    10,
    3,
    help="Simulated pitch, roll, and yaw vibration from a moving platform",
)
fov_deg = st.sidebar.slider(
    "Camera FOV (deg)",
    5.0,
    60.0,
    st.session_state.fov_deg,
    step=0.5,
    help="Total horizontal field of view used for azimuth conversion.",
)
st.session_state.fov_deg = fov_deg

simulate_cloud = st.sidebar.toggle(
    "Predict Cloud Occlusion",
    True,
    help="Simulate a moving cloud and warn before it blocks the beacon",
)
block_camera = st.sidebar.toggle(
    "Block Camera (Occlusion)",
    False,
    help="Simulate complete optical blockage",
)
simulate_decoy = st.sidebar.toggle(
    "Simulate Bird / Decoy",
    True,
    help="Inject occasional fast-moving false positives for rejection testing",
)

st.session_state.camera_mode = camera_mode

if st.session_state.session_running:
    if st.sidebar.button("Stop session", width="stretch"):
        st.session_state.session_running = False
        st.rerun()
else:
    if st.sidebar.button("Resume session", type="primary", width="stretch"):
        st.session_state.session_running = True
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🎯 System Status")
st.sidebar.metric(label="FOV", value=f"{fov_deg:.1f}°")
st.sidebar.metric(label="Live Feed Rate", value="5 FPS")
st.sidebar.caption("Controls update the live simulation immediately.")

telemetry_df = pd.DataFrame(st.session_state.telemetry_log)
st.sidebar.download_button(
    "📥 Download Mission Telemetry (CSV)",
    data=telemetry_df.to_csv(index=False),
    file_name="fsoc_mission_telemetry.csv",
    mime="text/csv",
    width="stretch",
    disabled=telemetry_df.empty,
)

overview_1, overview_2, overview_3, overview_4 = st.columns(4)
overview_1.metric("Field of View", f"{fov_deg:.1f}°")
overview_2.metric("Camera Resolution", "640 × 480")
overview_3.metric("Detector", detector.replace(" / Temporal", "").replace(" Simulation", ""))
overview_4.metric("ROI Window", "128 × 128")


def compute_angular_offsets(pred_x, pred_y, w=640, h=480, fov_deg=20.0):
    dx_px = pred_x - (w / 2.0)
    dy_px = pred_y - (h / 2.0)
    azimuth_deg = dx_px * (fov_deg / float(w))
    elevation_deg = dy_px * (fov_deg / float(h))
    return dx_px, dy_px, azimuth_deg, elevation_deg


def compute_mrad_error_from_angles(az_deg, el_deg):
    az_rad = np.deg2rad(abs(float(az_deg)))
    el_rad = np.deg2rad(abs(float(el_deg)))
    return float(np.sqrt(az_rad**2 + el_rad**2) * 1000.0)


def read_real_camera_frame():
    cap = st.session_state.camera_cap
    if cap is None:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None, None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        st.session_state.camera_cap = cap

    ok, frame = cap.read()
    if not ok:
        return None, None

    height, width = frame.shape[:2]
    if height > width:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    frame = cv2.flip(frame, 1)

    # Normalize the webcam image in Lab color space to make the laser dot more distinct
    # on a white plane without pushing a synthetic, over-sharpened contrast map into the feed.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_stretched = clahe.apply(l)
    lab = cv2.merge((l_stretched, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    detector = LaserSpotDetector(min_area=8, max_area=5000)
    result = detector.detect(frame)
    if result is None:
        return frame, None

    cx, cy = result["centroid"]

    if "laser_history" not in st.session_state:
        st.session_state.laser_history = deque(maxlen=5)
    st.session_state.laser_history.append((cx, cy))
    if len(st.session_state.laser_history) >= 2:
        xs = [p[0] for p in st.session_state.laser_history]
        ys = [p[1] for p in st.session_state.laser_history]
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))

    return frame, (cx, cy)


def build_frames(
    turbulence,
    target_speed,
    detector,
    receiver_shake,
    block_camera,
    simulate_cloud,
    motion_mode,
    manual_target_x=320,
    manual_target_y=240,
):
    st.session_state.time_step += 1
    width, height = 640, 480
    center_x, center_y = width // 2, height // 2
    kf = st.session_state.kalman

    if st.session_state.camera_mode == "REAL CAMERA MODE":
        frame, real_detection = read_real_camera_frame()
        if frame is None:
            raw_frame = np.zeros((height, width, 3), dtype=np.uint8)
            detected_x = detected_y = None
            true_x = center_x
            true_y = center_y
            pred_x = pred_y = center_x
            camera_occluded = True
            data_link_status = "Camera Not Ready"
            data_rate_gbps = 0.0
            error_mrad = 0.0
            telemetry = {
                "true_x": true_x,
                "true_y": true_y,
                "pred_x": pred_x,
                "pred_y": pred_y,
                "width": width,
                "height": height,
                "inference_mode": "Real Camera",
                "inference_fps": 0,
                "cloud_eta_ms": 0,
                "preparing_occlusion": False,
                "cloud_occluded": False,
                "shake_pitch": 0.0,
                "shake_roll": 0.0,
                "shake_yaw": 0.0,
                "confidence": 0.0,
                "decoy_active": False,
                "decoy_rejected": False,
                "data_rate_gbps": 0.0,
                "data_link_status": data_link_status,
                "azimuth_deg": 0.0,
                "elevation_deg": 0.0,
            }
            st.session_state.latest_frame_data = telemetry
            return raw_frame, raw_frame.copy()

        raw_frame = frame.copy()
        cv2.line(raw_frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 255, 255), 1)
        cv2.line(raw_frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 255, 255), 1)

        detected_x, detected_y = real_detection if real_detection is not None else (None, None)
        if detected_x is not None and detected_y is not None:
            cv2.circle(raw_frame, (int(detected_x), int(detected_y)), 10, (0, 255, 0), 2)
            cv2.putText(raw_frame, "LASER DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(raw_frame, "NO LASER DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if detected_x is not None and detected_y is not None:
            prev_pos = np.asarray(st.session_state.last_known_pos, dtype=float)
            measurement = np.array([detected_x, detected_y], dtype=float)
            pred = kf.update(measurement)
            pred_x, pred_y = pred
            new_pos = np.array([float(pred_x), float(pred_y)], dtype=float)
            st.session_state.velocity = tuple((new_pos - prev_pos).tolist())
            st.session_state.last_known_pos = (float(pred_x), float(pred_y))
            true_x = detected_x
            true_y = detected_y
            confidence = 0.92
            camera_occluded = False
        else:
            prev_pos = np.asarray(st.session_state.last_known_pos, dtype=float)
            pred = kf.predict()
            pred_x, pred_y = pred
            new_pos = np.array([float(pred_x), float(pred_y)], dtype=float)
            st.session_state.velocity = tuple((new_pos - prev_pos).tolist())
            st.session_state.last_known_pos = (float(pred_x), float(pred_y))
            true_x = center_x
            true_y = center_y
            confidence = 0.0
            camera_occluded = False

        pred_x = float(np.clip(pred_x, 0, width - 1))
        pred_y = float(np.clip(pred_y, 0, height - 1))
        dx_px, dy_px, azimuth_deg, elevation_deg = compute_angular_offsets(pred_x, pred_y, width, height, st.session_state.fov_deg)
        error_mrad = compute_mrad_error_from_angles(azimuth_deg, elevation_deg)
        data_rate_gbps = 10.0 if confidence > 0.5 else 0.0
        data_link_status = "Optimal" if data_rate_gbps > 0 else "Camera idle"
        st.session_state.latest_frame_data = {
            "true_x": true_x,
            "true_y": true_y,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "width": width,
            "height": height,
            "inference_mode": "Real Camera",
            "inference_fps": 15,
            "cloud_eta_ms": 0,
            "preparing_occlusion": False,
            "cloud_occluded": False,
            "shake_pitch": 0.0,
            "shake_roll": 0.0,
            "shake_yaw": 0.0,
            "confidence": confidence,
            "decoy_active": False,
            "decoy_rejected": False,
            "data_rate_gbps": data_rate_gbps,
            "data_link_status": data_link_status,
            "azimuth_deg": float(azimuth_deg),
            "elevation_deg": float(elevation_deg),
            "dx_px": float(dx_px),
            "dy_px": float(dy_px),
            "lock_state": "LOCKED" if confidence > 0.5 else "PREDICTIVE MODE",
        }
        st.session_state.error_history.append(float(error_mrad))
        st.session_state.time_history.append(st.session_state.time_step)
        if len(st.session_state.error_history) > 60:
            st.session_state.error_history.pop(0)
            st.session_state.time_history.pop(0)
        if len(st.session_state.telemetry_log) > 1000:
            st.session_state.telemetry_log.pop(0)
        st.session_state.telemetry_log.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "frame": st.session_state.time_step,
                "true_x": float(true_x),
                "true_y": float(true_y),
                "pred_x": float(pred_x),
                "pred_y": float(pred_y),
                "error_mrad": round(float(error_mrad), 4),
                "azimuth_deg": round(float(azimuth_deg), 4),
                "elevation_deg": round(float(elevation_deg), 4),
                "camera_mode": "REAL",
                "lock_state": "LOCKED" if confidence > 0.5 else "PREDICTIVE MODE",
            }
        )
        return raw_frame, raw_frame.copy()

    if motion_mode == "Manual Control":
        true_x = manual_target_x
        true_y = manual_target_y
    else:
        true_x = int(center_x + 150 * np.cos(st.session_state.time_step * (target_speed * 0.05)))
        true_y = int(center_y + 100 * np.sin(st.session_state.time_step * (target_speed * 0.05)))

    shake_pitch = receiver_shake * 2.5 * np.sin(st.session_state.time_step * 0.13)
    shake_roll = receiver_shake * 2.0 * np.cos(st.session_state.time_step * 0.11)
    shake_yaw = receiver_shake * 1.5 * np.sin(st.session_state.time_step * 0.08)
    apparent_x = int(np.clip(true_x + shake_yaw + shake_roll, 0, width - 1))
    apparent_y = int(np.clip(true_y + shake_pitch, 0, height - 1))

    raw_frame = np.zeros((height, width, 3), dtype=np.uint8)
    if not block_camera:
        # A laser-like point on a white plane is represented as a bright constant spot.
        cv2.circle(raw_frame, (apparent_x, apparent_y), 8, (255, 255, 255), -1)
        cv2.circle(raw_frame, (apparent_x, apparent_y), 15, (200, 200, 255), -1)
        if turbulence > 0:
            kernel_size = (turbulence * 2 + 1, turbulence * 2 + 1)
            raw_frame = cv2.GaussianBlur(raw_frame, kernel_size, 0)
            noise = np.random.normal(0, turbulence * 3, raw_frame.shape).astype(np.uint8)
            raw_frame = cv2.add(raw_frame, noise)

    cloud_x = int(-100 + ((st.session_state.time_step * 2.5) % (width + 200)))
    cloud_y = int(175 + 35 * np.sin(st.session_state.time_step * 0.035))
    cloud_left, cloud_top = cloud_x - 65, cloud_y - 45
    cloud_right, cloud_bottom = cloud_x + 65, cloud_y + 45
    cloud_inside = (
        cloud_left <= apparent_x <= cloud_right
        and cloud_top <= apparent_y <= cloud_bottom
    )
    cloud_distance = np.hypot(apparent_x - cloud_x, apparent_y - cloud_y)
    cloud_speed = max(1.0, 2.5)
    cloud_eta_ms = max(0, int((cloud_distance - 90) / cloud_speed * 200))
    preparing_occlusion = simulate_cloud and not cloud_inside and cloud_distance < 180
    cloud_occluded = simulate_cloud and cloud_inside
    camera_occluded = block_camera or cloud_occluded

    if simulate_cloud:
        cv2.rectangle(raw_frame, (cloud_left, cloud_top), (cloud_right, cloud_bottom), (70, 70, 70), -1)
        cv2.putText(raw_frame, "CLOUD", (max(5, cloud_left + 12), cloud_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1)

    cv2.line(raw_frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 255, 255), 1)
    cv2.line(raw_frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 255, 255), 1)
    cv2.circle(raw_frame, (center_x, center_y), 30, (0, 255, 255), 1)
    cv2.putText(raw_frame, f"IMU P:{shake_pitch:+.1f} R:{shake_roll:+.1f} Y:{shake_yaw:+.1f}", (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    ai_frame = raw_frame.copy()
    decoy_active = simulate_decoy and st.session_state.time_step % 37 < 5
    decoy_x = int((st.session_state.time_step * 19) % width)
    decoy_y = int(70 + 80 * np.sin(st.session_state.time_step * 0.22))
    if decoy_active and not camera_occluded:
        cv2.circle(raw_frame, (decoy_x, decoy_y), 10, (255, 180, 80), -1)
        cv2.putText(raw_frame, "DECOY", (max(4, decoy_x - 25), max(18, decoy_y - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 80), 1)
        ai_frame = raw_frame.copy()

    inference_mode = "Full Frame"
    inference_fps = 15
    detected_x, detected_y = None, None
    confidence = 0.0
    decoy_rejected = False

    if st.session_state.tracking_mode == "ROI Tracking" and st.session_state.roi_center:
        roi_x, roi_y = st.session_state.roi_center
        roi_half = 64
        roi_contains_target = (
            roi_x - roi_half <= apparent_x <= roi_x + roi_half
            and roi_y - roi_half <= apparent_y <= roi_y + roi_half
        )
        if roi_contains_target:
            inference_mode = "ROI 128x128"
            inference_fps = 120
        else:
            st.session_state.tracking_mode = "Acquisition"
            st.session_state.roi_center = None

    if not camera_occluded and detector != "Detector Disabled":
        jitter_x = int(np.random.normal(0, 1 + turbulence * 0.5))
        jitter_y = int(np.random.normal(0, 1 + turbulence * 0.5))

        candidate_x = decoy_x if decoy_active else apparent_x + jitter_x
        candidate_y = decoy_y if decoy_active else apparent_y + jitter_y
        last_x, last_y = st.session_state.last_known_pos
        velocity_x, velocity_y = st.session_state.velocity
        expected_x = last_x + velocity_x
        expected_y = last_y + velocity_y
        innovation_distance = np.hypot(candidate_x - expected_x, candidate_y - expected_y)
        decoy_rejected = decoy_active and innovation_distance > 70

        detected_x = candidate_x
        detected_y = candidate_y
        if decoy_rejected:
            detected_x = None
            detected_y = None

        confidence = max(0.40, min(0.99, 0.98 - turbulence * 0.05))
        if detected_x is not None and detected_y is not None:
            st.session_state.roi_center = (detected_x, detected_y)
            st.session_state.tracking_mode = "ROI Tracking"
            if detector == "YOLOv8 Beacon Simulation":
                box_size = 24
                cv2.rectangle(ai_frame, (detected_x - box_size, detected_y - box_size), (detected_x + box_size, detected_y + box_size), (0, 255, 0), 2)
                label = f"YOLOv8 Beacon: {confidence:.2f}"
                label_color = (0, 255, 0)
            else:
                cv2.circle(ai_frame, (detected_x, detected_y), 18, (0, 255, 255), 2)
                label = f"Laser Spot: {confidence:.2f}"
                label_color = (0, 255, 255)
        else:
            label = "DECOY REJECTED - TRACK LOCKED"
            label_color = (0, 165, 255)
        cv2.putText(ai_frame, label, (18, 62) if decoy_rejected else (detected_x - 42, detected_y - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
    else:
        message = "SIGNAL LOST - OCCLUDED" if camera_occluded else "DETECTOR DISABLED"
        cv2.putText(ai_frame, message, (center_x - 140, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if detected_x is not None and detected_y is not None:
        prev_pos = np.asarray(st.session_state.last_known_pos, dtype=float)
        measurement = np.array([detected_x, detected_y], dtype=float)
        pred = kf.update(measurement)
        pred_x, pred_y = pred
        new_pos = np.array([float(pred_x), float(pred_y)], dtype=float)
        st.session_state.velocity = tuple((new_pos - prev_pos).tolist())
        st.session_state.last_known_pos = (float(pred_x), float(pred_y))
    else:
        prev_pos = np.asarray(st.session_state.last_known_pos, dtype=float)
        pred = kf.predict()
        pred_x, pred_y = pred
        new_pos = np.array([float(pred_x), float(pred_y)], dtype=float)
        st.session_state.velocity = tuple((new_pos - prev_pos).tolist())
        st.session_state.last_known_pos = (float(pred_x), float(pred_y))

    pred_x = float(np.clip(pred_x, 0, width - 1))
    pred_y = float(np.clip(pred_y, 0, height - 1))

    dx_px, dy_px, azimuth_deg, elevation_deg = compute_angular_offsets(pred_x, pred_y, width, height, st.session_state.fov_deg)
    error_mrad = compute_mrad_error_from_angles(azimuth_deg, elevation_deg)

    if camera_occluded:
        data_rate_gbps = 0.0
        data_link_status = "Link Severed - Buffering Data"
    elif error_mrad < 0.5 and turbulence < 6:
        data_rate_gbps = 10.0
        data_link_status = "Optimal"
    elif error_mrad > 1.0 or turbulence >= 6:
        data_rate_gbps = 1.0
        data_link_status = "Degraded"
    else:
        data_rate_gbps = 5.0
        data_link_status = "Nominal"

    st.session_state.error_history.append(float(error_mrad))
    st.session_state.time_history.append(st.session_state.time_step)
    if len(st.session_state.error_history) > 60:
        st.session_state.error_history.pop(0)
        st.session_state.time_history.pop(0)

    if preparing_occlusion:
        cv2.putText(ai_frame, f"PREPARE FOR OCCLUSION ({cloud_eta_ms} ms)", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    if st.session_state.tracking_mode == "ROI Tracking" and st.session_state.roi_center:
        roi_x, roi_y = st.session_state.roi_center
        cv2.rectangle(ai_frame, (roi_x - 64, roi_y - 64), (roi_x + 64, roi_y + 64), (255, 120, 0), 1)
        cv2.putText(ai_frame, "ROI 128x128", (max(4, roi_x - 62), max(16, roi_y - 68)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)

    lock_state = "LOCKED" if confidence > 0.5 and not camera_occluded else "PREDICTIVE MODE" if not camera_occluded else "LOST"
    telemetry = {
        "true_x": float(true_x),
        "true_y": float(true_y),
        "pred_x": float(pred_x),
        "pred_y": float(pred_y),
        "width": width,
        "height": height,
        "inference_mode": inference_mode,
        "inference_fps": inference_fps,
        "cloud_eta_ms": cloud_eta_ms,
        "preparing_occlusion": preparing_occlusion,
        "cloud_occluded": cloud_occluded,
        "shake_pitch": float(shake_pitch),
        "shake_roll": float(shake_roll),
        "shake_yaw": float(shake_yaw),
        "confidence": float(confidence),
        "decoy_active": bool(decoy_active),
        "decoy_rejected": bool(decoy_rejected),
        "data_rate_gbps": float(data_rate_gbps),
        "data_link_status": data_link_status,
        "azimuth_deg": float(azimuth_deg),
        "elevation_deg": float(elevation_deg),
        "dx_px": float(dx_px),
        "dy_px": float(dy_px),
        "lock_state": lock_state,
    }
    st.session_state.latest_frame_data = telemetry

    if len(st.session_state.telemetry_log) > 1000:
        st.session_state.telemetry_log.pop(0)
    st.session_state.telemetry_log.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "frame": st.session_state.time_step,
            "true_x": round(float(true_x), 4),
            "true_y": round(float(true_y), 4),
            "pred_x": round(float(pred_x), 4),
            "pred_y": round(float(pred_y), 4),
            "error_mrad": round(float(error_mrad), 4),
            "azimuth_deg": round(float(azimuth_deg), 4),
            "elevation_deg": round(float(elevation_deg), 4),
            "cloud_occluded": cloud_occluded,
            "decoy_active": bool(decoy_active),
            "decoy_rejected": bool(decoy_rejected),
            "data_rate_gbps": round(float(data_rate_gbps), 4),
            "data_link_status": data_link_status,
        }
    )
    return raw_frame, ai_frame


@st.fragment(run_every=0.75)
def render_live_feeds():
    if not st.session_state.session_running:
        st.info("Session stopped. Select Resume session to restart live tracking.")
        return

    raw_frame, ai_frame = build_frames(
        turbulence,
        target_speed,
        detector,
        receiver_shake,
        block_camera,
        simulate_cloud,
        motion_mode,
        manual_target_x,
        manual_target_y,
    )
    telemetry = st.session_state.latest_frame_data
    st.markdown("## Live Optical Tracking")
    st.caption("Synthetic camera input and detector output update continuously.")

    lock_state = telemetry.get("lock_state", "LOCKED")
    state_class = "status-lock" if lock_state == "LOCKED" else "status-loss" if lock_state == "LOST" else "status-predictive"
    st.markdown(f'<div class="status-pill {state_class}">{lock_state}</div>', unsafe_allow_html=True)

    status_1, status_2, status_3, status_4, status_5 = st.columns(5)
    status_1.metric("Inference Path", telemetry["inference_mode"])
    status_2.metric("Estimated AI Rate", f'{telemetry["inference_fps"]} FPS')
    status_3.metric("Azimuth Offset", f'{telemetry["azimuth_deg"]:+.2f}°')
    status_4.metric("Elevation Offset", f'{telemetry["elevation_deg"]:+.2f}°')
    status_5.metric("Data Link", f'{telemetry["data_rate_gbps"]:.0f} Gbps')

    if telemetry["decoy_rejected"]:
        st.warning("DECOY REJECTED: impossible motion detected; target lock retained.")
    if telemetry["preparing_occlusion"]:
        st.warning(f"PREPARE FOR OCCLUSION: predicted cloud intersection in {telemetry['cloud_eta_ms']} milliseconds.")
    elif telemetry["cloud_occluded"]:
        st.error("CLOUD OCCLUSION ACTIVE: predictive tracking is maintaining the estimated position.")
    if lock_state == "LOST":
        st.error("LOCK LOST: sensor confidence dropped below threshold; Kalman filter is predicting the last valid state.")
    elif lock_state == "PREDICTIVE MODE":
        st.warning("PREDICTIVE MODE: tracking with motion model while waiting for a fresh beacon observation.")

    left_col, right_col = st.columns(2)
    with left_col:
        st.image(raw_frame, channels="BGR", caption="Synthetic camera input", width=640)
    with right_col:
        st.image(ai_frame, channels="BGR", caption="Detector + tracking output", width=640)


@st.fragment(run_every=3.0)
def render_tracking_charts():
    if not st.session_state.session_running:
        return

    telemetry = st.session_state.latest_frame_data
    if telemetry is None:
        return

    times = list(st.session_state.time_history)
    errors = list(st.session_state.error_history)
    min_len = min(len(times), len(errors))

    st.markdown("## Tracking Analysis")
    st.caption("Compare the predicted target against the real laser spot and track angular error in coarse-alignment coordinates.")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 3. EKF Tracking")
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatter(x=[telemetry["true_x"]], y=[telemetry["true_y"]], mode="markers", name="Ground Truth", marker=dict(color="green", size=12, symbol="circle")))
        fig_radar.add_trace(go.Scatter(x=[telemetry["pred_x"]], y=[telemetry["pred_y"]], mode="markers", name="EKF Estimate", marker=dict(color="red", size=14, symbol="x")))
        if telemetry["preparing_occlusion"]:
            fig_radar.add_annotation(x=telemetry["pred_x"], y=telemetry["pred_y"], text="Predictive mode armed", showarrow=True, arrowhead=2, font=dict(color="orange"))
        fig_radar.update_layout(
            xaxis=dict(range=[0, telemetry["width"]], title="Pixel X"),
            yaxis=dict(range=[telemetry["height"], 0], title="Pixel Y"),
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_radar, width="stretch")

    with col4:
        st.markdown("### 4. Coarse Alignment Error (mrad)")
        fig_error = go.Figure()
        if min_len > 0:
            smooth_errors = np.convolve(errors[:min_len], np.ones(5) / 5.0, mode="same")
            fig_error.add_trace(go.Scatter(x=times[:min_len], y=smooth_errors, mode="lines+markers", name="Error (mrad)", line=dict(color="orange", width=2)))
        fig_error.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Hand-off Threshold (1 mrad)")
        fig_error.update_layout(
            xaxis_title="Frame",
            yaxis_title="Error (mrad)",
            yaxis=dict(range=[0, max(2.0, max(np.nan_to_num(errors), default=1.0) + 0.5)]),
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_error, width="stretch")


render_live_feeds()
render_tracking_charts()
