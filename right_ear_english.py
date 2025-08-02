import cv2
import mediapipe as mp
import numpy as np
import logging
import time
from filterpy.kalman import KalmanFilter
import pyttsx3
import threading
import pygame
import queue

# Suppress MediaPipe warnings
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Initialize pygame mixer for beep sounds
pygame.mixer.init()
correct_beep = pygame.mixer.Sound('correct.wav')  # Path to correct pose beep sound
incorrect_beep = pygame.mixer.Sound('incorrect.wav')  # Path to incorrect pose beep sound

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.6,
    model_complexity=2,
    smooth_landmarks=True
)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
frame_width = 1280
frame_height = 720
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# --- Speech Queue System (Critical/Non-critical) ---
critical_speech_queue = queue.Queue()
noncritical_speech_queue = queue.Queue(maxsize=1)  # Only keep the latest non-critical

def speech_worker():
    while True:
        # Always process all critical instructions first
        while not critical_speech_queue.empty():
            text = critical_speech_queue.get()
            if text is None:
                return  # Allows for clean shutdown if needed
            engine.say(text)
            engine.runAndWait()
            critical_speech_queue.task_done()
        # Then process the latest non-critical instruction (if any)
        try:
            text = noncritical_speech_queue.get(timeout=0.1)
            if text is None:
                return
            engine.say(text)
            engine.runAndWait()
            noncritical_speech_queue.task_done()
        except queue.Empty:
            continue

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak(text, force=False, critical=False):
    if not hasattr(speak, 'last_text'):
        speak.last_text = None
    if speak.last_text != text or force:
        speak.last_text = text
        if critical:
            critical_speech_queue.put(text)
        else:
            # Only keep the latest non-critical instruction
            if not noncritical_speech_queue.empty():
                try:
                    noncritical_speech_queue.get_nowait()
                    noncritical_speech_queue.task_done()
                except queue.Empty:
                    pass
            noncritical_speech_queue.put(text)

# Define scaling factor for angles
ANGLE_SCALE = 1

# Initialize Kalman Filter for smoothing angles
def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.x = np.zeros(6)
    kf.F = np.array([
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])
    kf.P *= 10.
    kf.R = np.diag([1.0, 1.0, 1.0])
    kf.Q = np.eye(6) * 0.05
    return kf

kf = initialize_kalman_filter()

# Load target pose
target_pose = [
    {
        "person_id": 0,
        "bbox": [
            260.447998046875,
            434.9598693847656,
            263.357177734375,
            439.172119140625
        ],
        "keypoints": [
            {"name": "Nose", "x": 240.35791015625, "y": 135.41705322265625, "score": 0.9791688919067383},
            {"name": "L_Eye", "x": 265.16717529296875, "y": 110.43780517578125, "score": 0.9833072428857386},
            {"name": "R_Eye", "x": 210.517822265625, "y": 114.45855712890625, "score": 0.9687361121177673},
            {"name": "L_Ear", "x": 301.84814453125, "y": 135.83111572265625, "score": 0.9493670302238464},
            {"name": "R_Ear", "x": 175.035888671875, "y": 143.1534423828125, "score": 0.9537781476974487},
            {"name": "L_Shoulder", "x": 367.36688232421875, "y": 277.89508056640625, "score": 0.9714463949203491},
            {"name": "R_Shoulder", "x": 132.6015625, "y": 287.1273193359375, "score": 0.9208009243011475},
            {"name": "L_Elbow", "x": 404.8804931640625, "y": 457.8016357421875, "score": 1.0068358182907104},
            {"name": "R_Elbow", "x": 121.6767578125, "y": 466.985595703125, "score": 0.9445005059242249},
            {"name": "L_Wrist", "x": 316.5948486328125, "y": 564.1590576171875, "score": 0.9202994108200073},
            {"name": "R_Wrist", "x": 218.354248046875, "y": 578.4954833984375, "score": 0.9106894731521606},
            {"name": "L_Hip", "x": 343.258056640625, "y": 562.5377197265625, "score": 0.8454821705818176},
            {"name": "R_Hip", "x": 191.992431640625, "y": 569.1612548828125, "score": 0.856957733631134},
            {"name": "L_Knee", "x": 394.12591552734375, "y": 672.401611328125, "score": 0.8698152899742126},
            {"name": "R_Knee", "x": 143.781005859375, "y": 696.0062255859375, "score": 0.8501293659210205},
            {"name": "L_Ankle", "x": 353.07330322265625, "y": 853.671142578125, "score": 0.9136713147163391},
            {"name": "R_Ankle", "x": 211.80206298828125, "y": 850.3348388671875, "score": 0.8354711532592773}
        ]
    }
]

# Extract and center target keypoints
target_keypoints = [(kp["x"], kp["y"]) for kp in target_pose[0]["keypoints"]]
head_keypoint_indices = [0, 1, 2, 3, 4]
head_keypoints = [target_keypoints[i] for i in head_keypoint_indices]
target_head_center_x = sum(x for x, y in head_keypoints) / len(head_keypoints)
target_head_center_y = sum(y for x, y in head_keypoints) / len(head_keypoints)
display_center_x = frame_width / 2
display_center_y = frame_height * 0.2
translate_x = display_center_x - target_head_center_x
translate_y = display_center_y - target_head_center_y
centered_target_keypoints = [(x + translate_x, y + translate_y) for x, y in target_keypoints]
head_keypoints_centered = [centered_target_keypoints[i] for i in head_keypoint_indices]
x_coords = [x for x, y in head_keypoints_centered]
y_coords = [y for x, y in head_keypoints_centered]
bbox_min_x = max(0, min(x_coords) - 20)
bbox_max_x = min(frame_width, max(x_coords) + 20)
bbox_min_y = max(0, min(y_coords) - 20)
bbox_max_y = min(frame_height, max(y_coords) + 20)

# Helper functions
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_head_pose_matched(user_landmarks, target_keypoints, distance_threshold=25):
    head_indices_mapping = {0: 0, 2: 1, 5: 2, 7: 3, 8: 4}
    for mp_idx, target_idx in head_indices_mapping.items():
        if mp_idx < len(user_landmarks) and target_idx < len(target_keypoints):
            distance = euclidean_distance(user_landmarks[mp_idx], target_keypoints[target_idx])
            if distance > distance_threshold:
                return False
    return True

def is_full_body_visible(landmarks, frame_width, frame_height):
    key_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
    ]
    for landmark in key_landmarks:
        lm = landmarks[landmark]
        if (lm.visibility < 0.6 or
            lm.x < 0.05 or lm.x > 0.95 or
            lm.y < 0.05 or lm.y > 0.95):
            return False
    return True

def _calculate_raw_head_angles_user_method(landmark_list):
    required_indices = [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR,
                        mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE]
    if landmark_list is None or len(landmark_list) <= max(idx.value for idx in required_indices):
        return None
    for l_idx_enum in required_indices:
        if landmark_list[l_idx_enum.value].visibility < 0.5:
            return None
    nose = landmark_list[mp_pose.PoseLandmark.NOSE.value]
    left_ear = landmark_list[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmark_list[mp_pose.PoseLandmark.RIGHT_EAR.value]
    left_eye = landmark_list[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmark_list[mp_pose.PoseLandmark.RIGHT_EYE.value]
    mid_ear = np.array([(left_ear.x + right_ear.x) / 2,
                        (left_ear.y + right_ear.y) / 2,
                        (left_ear.z + right_ear.z) / 2])
    nose_vec = mid_ear - np.array([nose.x, nose.y, nose.z])
    yaw = -np.degrees(np.arctan2(nose_vec[0], nose_vec[2] + 1e-6))
    eye_mid = np.array([(left_eye.x + right_eye.x) / 2,
                        (left_eye.y + right_eye.y) / 2,
                        (left_eye.z + right_eye.z) / 2])
    nose_to_eye = np.array([nose.x, nose.y, nose.z]) - eye_mid
    pitch = np.degrees(np.arctan2(nose_to_eye[1], np.sqrt(nose_to_eye[0]**2 + nose_to_eye[2]**2 + 1e-6)))
    ear_vec_2d = np.array([left_ear.x - right_ear.x, left_ear.y - right_ear.y])
    roll = np.degrees(np.arctan2(ear_vec_2d[1], ear_vec_2d[0] + 1e-6))
    return yaw, -(pitch - 50), roll

def get_head_angles(pose_results):
    raw_yaw, raw_pitch, raw_roll = 0.0, 0.0, 0.0
    if pose_results and pose_results.pose_landmarks:
        try:
            angles = _calculate_raw_head_angles_user_method(
                pose_results.pose_landmarks.landmark
            )
            if angles is not None:
                raw_yaw, raw_pitch, raw_roll = angles
        except Exception as e:
            logging.error(f"Error in get_head_angles: {e}")
    kf.predict()
    kf.update(np.array([raw_yaw, raw_pitch, raw_roll]))
    smoothed_yaw, smoothed_pitch, smoothed_roll = kf.x[:3]
    return smoothed_yaw * ANGLE_SCALE * 3, smoothed_pitch * ANGLE_SCALE, smoothed_roll * ANGLE_SCALE

def wrap_angle_180(angle):
    wrapped_angle = np.fmod(angle + 180, 360)
    if wrapped_angle < 0:
        wrapped_angle += 360
    return wrapped_angle - 180


def run():
    # Timer and state variables (unchanged)
    visibility_confirmed = False
    match_start_time = None
    match_duration_threshold = 5  # Calibration hold time
    pose_held = False
    bppv_step_1 = False
    bppv_step_2 = False
    bppv_step_3 = False
    bppv_step_4 = False
    bppv_start_time = None
    bppv_duration_threshold = 45  # 45 seconds for BPPV steps 1-3
    neutral_hold_threshold = 5  # 5 seconds for neutral position in step 4
    bppv_pose_held_time = 0
    mission_complete = False
    step_3_complete = False
    all_missions_complete = False
    last_speech_time = 0
    speech_interval = 3  # Minimum interval between voice instructions (seconds)
    in_correct_pose_step_1 = False
    in_correct_pose_step_2 = False
    in_correct_pose_step_3 = False
    in_correct_pose_step_4 = False
    was_in_correct_pose_step_1 = False
    was_in_correct_pose_step_2 = False
    was_in_correct_pose_step_3 = False
    was_in_correct_pose_step_4 = False
    # BPPV Step 1: Yaw 45° left (±20°)
    target_yaw_min_step_1 = -65
    target_yaw_max_step_1 = -25
    # BPPV Step 2: Yaw -25°, Pitch 90°, Roll 85° (±20°)
    target_yaw_min_step_2 = -35
    target_yaw_max_step_2 = 5
    target_pitch_min_step_2 = 75
    target_pitch_max_step_2 = 115
    target_roll_min_step_2 = 65
    target_roll_max_step_2 = 105
    # BPPV Step 3: Yaw -165°, Pitch 37°, Roll -100° (±20°)
    target_yaw_min_step_3 = -179
    target_yaw_max_step_3 = -150
    target_pitch_min_step_3 = 17
    target_pitch_max_step_3 = 57
    target_roll_min_step_3 = -120
    target_roll_max_step_3 = -80
    # BPPV Step 4: Neutral position opposite side(Yaw -180°, Pitch 10°, Roll -180° ±20°)
    target_yaw_min_step_4 = -160
    target_yaw_max_step_4 = 160
    target_pitch_min_step_4 = -10
    target_pitch_max_step_4 = 30
    target_roll_min_step_4 = -160
    target_roll_max_step_4 = 160

    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            speak("Camera is not open", critical=True)  # Critical instruction
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (frame_width, frame_height))
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image_rgb)
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        current_head_yaw, current_head_pitch, current_head_roll = 0, 0, 0
        current_time = time.time()

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            user_landmarks = [(lm.x * frame_width, lm.y * frame_height) for lm in landmarks]

            # Stage 1: Full-body visibility check
            if not visibility_confirmed:
                if is_full_body_visible(landmarks, frame_width, frame_height):
                    visibility_confirmed = True
                    cv2.putText(frame, "Visibility Confirmed!", (frame_width // 4, frame_height // 2 - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    if current_time - last_speech_time > speech_interval:
                        # Critical instruction to ensure it's always spoken
                        speak("Full body visibility confirmed. Please adjust your head to match the position that your eye and nose point are fully inside the box and box should be green", critical=True)
                        last_speech_time = current_time
                else:
                    cv2.putText(frame, "Please move back for full body visibility", (frame_width // 4 - 50, frame_height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                    if current_time - last_speech_time > speech_interval:
                        speak("Please move back to ensure your full body is visible in the frame.")
                        last_speech_time = current_time
                    match_start_time = None
                    pose_held = False
                    bppv_step_1 = False
                    bppv_step_2 = False
                    bppv_step_3 = False
                    bppv_step_4 = False
                    mission_complete = False
                    step_3_complete = False
                    all_missions_complete = False
                    in_correct_pose_step_1 = False
                    in_correct_pose_step_2 = False
                    in_correct_pose_step_3 = False
                    in_correct_pose_step_4 = False
                    was_in_correct_pose_step_1 = False
                    was_in_correct_pose_step_2 = False
                    was_in_correct_pose_step_3 = False
                    was_in_correct_pose_step_4 = False

            # Stage 2: Head pose matching and calibration
            elif visibility_confirmed and not pose_held:
                head_pose_matched = is_head_pose_matched(user_landmarks, centered_target_keypoints)
                bbox_color = (0, 255, 0) if head_pose_matched else (0, 0, 255)
                cv2.rectangle(frame, (int(bbox_min_x), int(bbox_min_y)), (int(bbox_max_x), int(bbox_max_y)),
                            bbox_color, 2)

                if head_pose_matched:
                    if match_start_time is None:
                        match_start_time = time.time()
                        if current_time - last_speech_time > speech_interval:
                            speak("Hold your head in this position.")
                            last_speech_time = current_time
                    else:
                        elapsed_time = time.time() - match_start_time
                        if elapsed_time >= match_duration_threshold:
                            pose_held = True
                            bppv_step_1 = True
                            correct_beep.play()
                            # Critical instruction to ensure it's always spoken
                            speak("Calibration complete. Now turn your head 45 degrees to the left and hold for 45 seconds.", critical=True)
                            last_speech_time = current_time
                            bppv_start_time = current_time
                        else:
                            remaining_time = max(0, match_duration_threshold - elapsed_time)
                            cv2.putText(frame, f"Hold Head Pose for {remaining_time:.1f}s",
                                        (frame_width // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    match_start_time = None
                    if current_time - last_speech_time > speech_interval:
                        incorrect_beep.play()
                        speak("Adjust your head to make the box green for 5 seconds.")
                        last_speech_time = current_time
                    cv2.putText(frame, "Adjust eye and nose in the centre of box", (frame_width // 4, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Stage 3: BPPV Step 1 - Turn head 45 degrees left and hold
            elif pose_held and bppv_step_1 and not mission_complete:
                current_head_yaw, current_head_pitch, current_head_roll = get_head_angles(pose_results)
                display_yaw = wrap_angle_180(current_head_yaw)
                display_pitch = wrap_angle_180(current_head_pitch)
                display_roll = wrap_angle_180(current_head_roll)

                yaw_correct = target_yaw_min_step_1 <= display_yaw <= target_yaw_max_step_1

                if yaw_correct:
                    if not in_correct_pose_step_1:
                        correct_beep.play()
                        in_correct_pose_step_1 = True
                        was_in_correct_pose_step_1 = True
                        speak("Hold this position for 45 seconds.")
                        last_speech_time = current_time
                    if bppv_start_time is None:
                        bppv_start_time = current_time
                    bppv_pose_held_time = current_time - bppv_start_time
                    remaining_time = max(0, bppv_duration_threshold - bppv_pose_held_time)
                    cv2.putText(frame, f"Hold Head at this position for {remaining_time:.1f}s",
                                (frame_width // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if current_time - last_speech_time > speech_interval:
                        speak(f"Hold your head at this position {remaining_time:.1f} seconds remaining.")
                        last_speech_time = current_time
                    if bppv_pose_held_time >= bppv_duration_threshold:
                        mission_complete = True
                        bppv_step_2 = True
                        correct_beep.play()
                        # Critical instruction to ensure it's always spoken
                        speak("Now, slowly lie down on your left side with your head facing the ceiling. Hold this pose for 45 seconds and stay relaxed.", critical=True)
                        last_speech_time = current_time
                        bppv_start_time = None
                        bppv_pose_held_time = 0
                        in_correct_pose_step_1 = False
                        was_in_correct_pose_step_1 = False
                else:
                    if was_in_correct_pose_step_1:
                        incorrect_beep.play()
                        was_in_correct_pose_step_1 = False
                    in_correct_pose_step_1 = False
                    bppv_start_time = None
                    if display_yaw < target_yaw_min_step_1:
                        if current_time - last_speech_time > speech_interval:
                            speak("Turn your head further to the right.")
                            last_speech_time = current_time
                        cv2.putText(frame, "Turn head further right", (frame_width // 4, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif display_yaw > target_yaw_max_step_1:
                        if current_time - last_speech_time > speech_interval:
                            speak("Turn your head back to the left.")
                            last_speech_time = current_time
                        cv2.putText(frame, "Turn head back left", (frame_width // 4, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.putText(frame, f"Yaw: {int(display_yaw)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, f"Pitch: {int(display_pitch)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, f"Roll: {int(display_roll)}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            # Stage 4: BPPV Step 2 - Yaw 0°, Pitch 90°, Roll -100°
            elif mission_complete and bppv_step_2 and not step_3_complete:
                current_head_yaw, current_head_pitch, current_head_roll = get_head_angles(pose_results)
                display_yaw = wrap_angle_180(current_head_yaw)
                display_pitch = wrap_angle_180(current_head_pitch)
                display_roll = wrap_angle_180(current_head_roll)

                yaw_correct = target_yaw_min_step_2 <= display_yaw <= target_yaw_max_step_2
                pitch_correct = target_pitch_min_step_2 <= display_pitch <= target_pitch_max_step_2
                roll_correct = target_roll_min_step_2 <= display_roll <= target_roll_max_step_2
                pose_correct = yaw_correct and pitch_correct and roll_correct

                if pose_correct:
                    if not in_correct_pose_step_2:
                        correct_beep.play()
                        in_correct_pose_step_2 = True
                        was_in_correct_pose_step_2 = True
                        speak("Hold this position for 45 seconds.")
                        last_speech_time = current_time
                    if bppv_start_time is None:
                        bppv_start_time = current_time
                    bppv_pose_held_time = current_time - bppv_start_time
                    remaining_time = max(0, bppv_duration_threshold - bppv_pose_held_time)
                    cv2.putText(frame, f"Hold Head at this position for {remaining_time:.1f}s",
                                (frame_width // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if current_time - last_speech_time > speech_interval:
                        speak(f"Hold your head in this position. {remaining_time:.1f} seconds remaining.")
                        last_speech_time = current_time
                    if bppv_pose_held_time >= bppv_duration_threshold:
                        step_3_complete = True
                        bppv_step_3 = True
                        correct_beep.play()
                        # Critical instruction to ensure it's always spoken
                        speak("Step 2 complete. Stay your head at the same angle, and roll your body to the left that your head facing the bed and hold for 45 seconds.", critical=True)
                        last_speech_time = current_time
                        bppv_start_time = None
                        bppv_pose_held_time = 0
                        in_correct_pose_step_2 = False
                        was_in_correct_pose_step_2 = False
                else:
                    if was_in_correct_pose_step_2:
                        incorrect_beep.play()
                        was_in_correct_pose_step_2 = False
                    in_correct_pose_step_2 = False
                    bppv_start_time = None
                    error_messages = []
                    if not yaw_correct:
                        if display_yaw < target_yaw_min_step_2:
                            error_messages.append("Turn your head to the right.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Turn your head to the right.")
                                last_speech_time = current_time
                        elif display_yaw > target_yaw_max_step_2:
                            error_messages.append("Turn your head to the left.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Turn your head to the left.")
                                last_speech_time = current_time
                    if not pitch_correct:
                        if display_pitch < target_pitch_min_step_2:
                            error_messages.append("Tilt your head further up.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Tilt your head further up.")
                                last_speech_time = current_time
                        elif display_pitch > target_pitch_max_step_2:
                            error_messages.append("Tilt your head down.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Tilt your head down.")
                                last_speech_time = current_time
                    if not roll_correct:
                        if display_roll < target_roll_min_step_2:
                            error_messages.append("Bend your head more to the right.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Bend your head more to the right.")
                                last_speech_time = current_time
                        elif display_roll > target_roll_max_step_2:
                            error_messages.append("Bend your head to the left.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Bend your head to the left.")
                                last_speech_time = current_time
                    error_text = " ".join(error_messages) if error_messages else "Adjust head to target pose."
                    cv2.putText(frame, error_text, (frame_width // 4, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.putText(frame, f"Yaw: {int(display_yaw)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, f"Pitch: {int(display_pitch)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, f"Roll: {int(display_roll)}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Stage 5: BPPV Step 3 - Yaw 173°, Pitch 37°, Roll 97°
            elif step_3_complete and bppv_step_3 and not bppv_step_4:
                current_head_yaw, current_head_pitch, current_head_roll = get_head_angles(pose_results)
                display_yaw = wrap_angle_180(current_head_yaw)
                display_pitch = wrap_angle_180(current_head_pitch)
                display_roll = wrap_angle_180(current_head_roll)

                yaw_correct = target_yaw_min_step_3 <= display_yaw <= target_yaw_max_step_3
                pitch_correct = target_pitch_min_step_3 <= display_pitch <= target_pitch_max_step_3
                roll_correct = target_roll_min_step_3 <= display_roll <= target_roll_max_step_3
                pose_correct = yaw_correct and pitch_correct and roll_correct

                if pose_correct:
                    if not in_correct_pose_step_3:
                        correct_beep.play()
                        in_correct_pose_step_3 = True
                        was_in_correct_pose_step_3 = True
                        speak("Hold this position for 45 seconds.")
                        last_speech_time = current_time
                    if bppv_start_time is None:
                        bppv_start_time = current_time
                    bppv_pose_held_time = current_time - bppv_start_time
                    remaining_time = max(0, bppv_duration_threshold - bppv_pose_held_time)
                    cv2.putText(frame, f"Hold Head at this position for {remaining_time:.1f}s",
                                (frame_width // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if current_time - last_speech_time > speech_interval:
                        speak(f"Hold your head in this position. {remaining_time:.1f} seconds remaining.")
                        last_speech_time = current_time
                    if bppv_pose_held_time >= bppv_duration_threshold:
                        bppv_step_4 = True
                        correct_beep.play()
                        # Critical instruction to ensure it's always spoken
                        speak("Step 3 complete. Shake your head 2 to 3 times and then sit on the opposite side of the bed in a neutral position for 5 seconds.", critical=True)
                        last_speech_time = current_time
                        bppv_start_time = None
                        bppv_pose_held_time = 0
                        in_correct_pose_step_3 = False
                        was_in_correct_pose_step_3 = False
                else:
                    if was_in_correct_pose_step_3:
                        incorrect_beep.play()
                        was_in_correct_pose_step_3 = False
                    in_correct_pose_step_3 = False
                    bppv_start_time = None
                    error_messages = []
                    if not yaw_correct:
                        if display_yaw < target_yaw_min_step_3:
                            error_messages.append("Turn your head further to the right.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Turn your head further to the right.")
                                last_speech_time = current_time
                        elif display_yaw > target_yaw_max_step_3:
                            error_messages.append("Turn your head back to the left.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Turn your head back to the left.")
                                last_speech_time = current_time
                    if not pitch_correct:
                        if display_pitch < target_pitch_min_step_3:
                            error_messages.append("Tilt your head further up.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Tilt your head further up.")
                                last_speech_time = current_time
                        elif display_pitch > target_pitch_max_step_3:
                            error_messages.append("Tilt your head down.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Tilt your head down.")
                                last_speech_time = current_time
                    if not roll_correct:
                        if display_roll < target_roll_min_step_3:
                            error_messages.append("Bend your head more to the left.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Bend your head more to the left.")
                                last_speech_time = current_time
                        elif display_roll > target_roll_max_step_3:
                            error_messages.append("Bend your head to the right.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Bend your head to the right.")
                                last_speech_time = current_time
                    error_text = " ".join(error_messages) if error_messages else "Adjust head to target pose."
                    cv2.putText(frame, error_text, (frame_width // 4, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.putText(frame, f"Yaw: {int(display_yaw)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, f"Pitch: {int(display_pitch)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, f"Roll: {int(display_roll)}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Stage 6: BPPV Step 4 - Sit in neutral position
            elif bppv_step_4 and not all_missions_complete:
                current_head_yaw, current_head_pitch, current_head_roll = get_head_angles(pose_results)
                display_yaw = wrap_angle_180(current_head_yaw)
                display_pitch = wrap_angle_180(current_head_pitch)
                display_roll = wrap_angle_180(current_head_roll)

                yaw_correct = (display_yaw < target_yaw_min_step_4) or (display_yaw > target_yaw_max_step_4)
                pitch_correct = (target_pitch_min_step_4 <= display_pitch <= target_pitch_max_step_4)
                roll_correct = (display_roll < target_roll_min_step_4) or (display_roll > target_roll_max_step_4)
                pose_correct = yaw_correct and pitch_correct and roll_correct

                if pose_correct:
                    if not in_correct_pose_step_4:
                        speak("Hold this neutral position for 5 seconds.")
                        last_speech_time = current_time
                        in_correct_pose_step_4 = True
                    if bppv_start_time is None:
                        bppv_start_time = current_time
                    bppv_pose_held_time = current_time - bppv_start_time
                    remaining_time = max(0, neutral_hold_threshold - bppv_pose_held_time)
                    cv2.putText(frame, f"Hold Neutral Position for {remaining_time:.1f}s",
                                (frame_width // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if current_time - last_speech_time > speech_interval:
                        speak(f"Hold your head at this position {remaining_time:.1f} seconds remaining.")
                        last_speech_time = current_time
                    if bppv_pose_held_time >= neutral_hold_threshold:
                        all_missions_complete = True
                        print("BPPV Step 4 Complete!")
                        # Critical instruction to ensure it's always spoken
                        speak("You have successfully completed the maneuver.", critical=True)
                        last_speech_time = current_time
                else:
                    bppv_start_time = None
                    in_correct_pose_step_4 = False
                    error_messages = []
                    if not yaw_correct:
                        if display_yaw > target_yaw_min_step_4:
                            error_messages.append("Turn your head further right.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Turn your head further right.")
                                last_speech_time = current_time
                        elif display_yaw < target_yaw_max_step_4:
                            error_messages.append("Turn your head further left.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Turn your head further left.")
                                last_speech_time = current_time
                    if not pitch_correct:
                        if display_pitch < target_pitch_min_step_4:
                            error_messages.append("Raise your head.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Raise your head.")
                                last_speech_time = current_time
                        elif display_pitch > target_pitch_max_step_4:
                            error_messages.append("Lower your head.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Lower your head.")
                                last_speech_time = current_time
                    if not roll_correct:
                        if display_roll > target_roll_min_step_4:
                            error_messages.append("Tilt your head to the left.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Tilt your head to the left.")
                                last_speech_time = current_time
                        elif display_roll < target_roll_max_step_4:
                            error_messages.append("Tilt your head to the right.")
                            if current_time - last_speech_time > speech_interval:
                                speak("Tilt your head to the right.")
                                last_speech_time = current_time
                    error_text = " ".join(error_messages) if error_messages else "Adjust to neutral position."
                    cv2.putText(frame, error_text, (frame_width // 4, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.putText(frame, f"Yaw: {int(display_yaw)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, f"Pitch: {int(display_pitch)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, f"Roll: {int(display_roll)}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Stage 7: All Missions Complete
            elif all_missions_complete:
                cv2.putText(frame, "GANS repositioning maneuver Complete!", (frame_width // 4, frame_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                current_head_yaw, current_head_pitch, current_head_roll = get_head_angles(pose_results)
                display_yaw = wrap_angle_180(current_head_yaw)
                display_pitch = wrap_angle_180(current_head_pitch)
                display_roll = wrap_angle_180(current_head_roll)
                cv2.putText(frame, f"Yaw: {int(display_yaw)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, f"Pitch: {int(display_pitch)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, f"Roll: {int(display_roll)}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

        cv2.imshow('AI Based BPPV Maneuver Guider', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup (unchanged)
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    pygame.mixer.quit()
    critical_speech_queue.put(None)
    noncritical_speech_queue.put(None)
    speech_thread.join(timeout=1)

if __name__ == "__main__":
    run()