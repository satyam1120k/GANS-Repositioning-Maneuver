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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress MediaPipe warnings
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Initialize pygame mixer for beep sounds
pygame.mixer.init()
correct_beep = pygame.mixer.Sound('correct.wav')  # Ensure these files exist
incorrect_beep = pygame.mixer.Sound('incorrect.wav')

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
    logging.error("Could not open camera.")
    print("Error: Could not open camera.")
    exit()

# Initialize pyttsx3 for text-to-speech
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    # Check for Hindi voice
    voices = engine.getProperty('voices')
    hindi_voice_set = False
    for voice in voices:
        if 'hindi' in voice.name.lower() or 'hi' in voice.languages:
            engine.setProperty('voice', voice.id)
            hindi_voice_set = True
            logging.info(f"Hindi voice set: {voice.name}")
            break
    if not hindi_voice_set:
        logging.warning("Hindi voice is not available. Using default voice.")
        print("Hindi voice is not available. Using default voice.")
    PYTTSX3_AVAILABLE = True
except Exception as e:
    logging.error(f"Failed to initialize pyttsx3: {e}")
    print("Error: Failed to initialize text-to-speech engine.")
    PYTTSX3_AVAILABLE = False
    engine = None
    hindi_voice_set = False

# --- Speech Queue System ---
critical_speech_queue = queue.Queue()
noncritical_speech_queue = queue.Queue(maxsize=1)

def speech_worker():
    while True:
        try:
            # Process critical instructions first
            while not critical_speech_queue.empty():
                text = critical_speech_queue.get()
                if text is None:
                    return
                if PYTTSX3_AVAILABLE and engine:
                    engine.say(text)
                    engine.runAndWait()
                    logging.info(f"Spoke: {text}")
                critical_speech_queue.task_done()
            # Process latest non-critical instruction
            try:
                text = noncritical_speech_queue.get(timeout=0.1)
                if text is None:
                    return
                if PYTTSX3_AVAILABLE and engine:
                    engine.say(text)
                    engine.runAndWait()
                    # logging.info(f"Spoke: {text}")
                noncritical_speech_queue.task_done()
            except queue.Empty:
                continue
        except Exception as e:
            logging.error(f"Error in speech_worker: {e}")
            time.sleep(0.1)

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
    # Timer and state variables
    visibility_confirmed = False
    match_start_time = None
    match_duration_threshold = 5
    pose_held = False
    bppv_step_1 = False
    bppv_step_2 = False
    bppv_step_3 = False
    bppv_step_4 = False
    bppv_start_time = None
    bppv_duration_threshold = 45
    neutral_hold_threshold = 5  # Fixed to 5 seconds as per instruction
    bppv_pose_held_time = 0
    mission_complete = False
    step_3_complete = False
    all_missions_complete = False
    last_speech_time = 0
    speech_interval = 3
    in_correct_pose_step_1 = False
    in_correct_pose_step_2 = False
    in_correct_pose_step_3 = False
    in_correct_pose_step_4 = False
    was_in_correct_pose_step_1 = False
    was_in_correct_pose_step_2 = False
    was_in_correct_pose_step_3 = False
    was_in_correct_pose_step_4 = False
    target_yaw_min_step_1 = 25
    target_yaw_max_step_1 = 65
    target_yaw_min_step_2 = -20
    target_yaw_max_step_2 = 20
    target_pitch_min_step_2 = 60
    target_pitch_max_step_2 = 110
    target_roll_min_step_2 = -120
    target_roll_max_step_2 = -80
    target_yaw_min_step_3 = 153
    target_yaw_max_step_3 = 193
    target_pitch_min_step_3 = 17
    target_pitch_max_step_3 = 57
    target_roll_min_step_3 = 77
    target_roll_max_step_3 = 117
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
            speak("कैमरा खुला नहीं है।", critical=True)
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (frame_width, frame_height))
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image_rgb)
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Display "Hindi voice is not available" on frame if applicable
        if not hindi_voice_set:
            cv2.putText(frame, "Hindi voice is not available. Using default voice.",
                        (frame_width // 4, frame_height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

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
                        speak("पूरा शरीर दिख रहा है । कृपया अपने सिर को इस तरह समायोजित करें कि आपकी आँख और नाक बिंदु पूरी तरह से बॉक्स के अंदर हों और बॉक्स हरा होना चाहिए।", critical=True)
                        last_speech_time = current_time
                else:
                    cv2.putText(frame, "Please move back for full body visibility", (frame_width // 4 - 50, frame_height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                    if current_time - last_speech_time > speech_interval:
                        speak("कृपया पीछे हटें ताकि आपका पूरा शरीर फ्रेम में दिखाई दे।")
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
                            speak("अपने सिर को इस स्थिति में बनाये रखें।")
                            last_speech_time = current_time
                    else:
                        elapsed_time = time.time() - match_start_time
                        if elapsed_time >= match_duration_threshold:
                            pose_held = True
                            bppv_step_1 = True
                            correct_beep.play()
                            speak("कैलिब्रेशन पूरा हुआ। अब अपने सिर को 45 डिग्री दाईं ओर मोड़ें और 45 सेकंड तक बनाये रखें।", critical=True)
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
                        speak("अपने सिर को समायोजित करें ताकि बॉक्स 5 सेकंड के लिए हरा हो।")
                        last_speech_time = current_time
                    cv2.putText(frame, "Adjust eye and nose in the centre of box", (frame_width // 4, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Stage 3: BPPV Step 1 - Turn head 45 degrees right and hold
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
                        speak("इस स्थिति को 45 सेकंड तक बनाये रखें।")
                        last_speech_time = current_time
                    if bppv_start_time is None:
                        bppv_start_time = current_time
                    bppv_pose_held_time = current_time - bppv_start_time
                    remaining_time = max(0, bppv_duration_threshold - bppv_pose_held_time)
                    cv2.putText(frame, f"Hold Head at this position for {remaining_time:.1f}s",
                                (frame_width // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if current_time - last_speech_time > speech_interval:
                        speak(f"अपने सिर को इस स्थिति में {remaining_time:.1f} सेकंड तक रखें।")
                        last_speech_time = current_time
                    if bppv_pose_held_time >= bppv_duration_threshold:
                        mission_complete = True
                        bppv_step_2 = True
                        correct_beep.play()
                        speak("अब, धीरे-धीरे अपनी बाईं ओर लेट जाएं, सिर को छत की ओर रखते हुए। इस मुद्रा को 45 सेकंड तक रखें और आराम करें।", critical=True)
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
                            speak("अपने सिर को और दाईं ओर मोड़ें।")
                            last_speech_time = current_time
                        cv2.putText(frame, "Turn head further right", (frame_width // 4, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif display_yaw > target_yaw_max_step_1:
                        if current_time - last_speech_time > speech_interval:
                            speak("अपने सिर को बाईं ओर वापस मोड़ें।")
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
                        speak("इस स्थिति को 45 सेकंड तक रखें।")
                        last_speech_time = current_time
                    if bppv_start_time is None:
                        bppv_start_time = current_time
                    bppv_pose_held_time = current_time - bppv_start_time
                    remaining_time = max(0, bppv_duration_threshold - bppv_pose_held_time)
                    cv2.putText(frame, f"Hold Head at this position for {remaining_time:.1f}s",
                                (frame_width // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if current_time - last_speech_time > speech_interval:
                        speak(f"अपने सिर को इस स्थिति में {remaining_time:.1f} सेकंड तक रखें।")
                        last_speech_time = current_time
                    if bppv_pose_held_time >= bppv_duration_threshold:
                        step_3_complete = True
                        bppv_step_3 = True
                        correct_beep.play()
                        speak("दूसरा चरण पूरा। अपने सिर को उसी कोण पर रखें, और अपने शरीर को दाईं ओर लुढ़काएं ताकि आपका सिर बिस्तर की ओर हो और 45 सेकंड तक रखें।", critical=True)
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
                            error_messages.append("अपने सिर को बाईं ओर मोड़ें।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपने सिर को बाईं ओर मोड़ें।")
                                last_speech_time = current_time
                        elif display_yaw > target_yaw_max_step_2:
                            error_messages.append("अपने सिर को दाईं ओर मोड़ें।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपने सिर को दाईं ओर मोड़ें।")
                                last_speech_time = current_time
                    if not pitch_correct:
                        if display_pitch < target_pitch_min_step_2:
                            error_messages.append("अपने सिर को और ऊपर झुकाएं।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपने सिर को और ऊपर झुकाएं।")
                                last_speech_time = current_time
                        elif display_pitch > target_pitch_max_step_2:
                            error_messages.append("अपने सिर को नीचे झुकाएं।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपने सिर को नीचे झुकाएं।")
                                last_speech_time = current_time
                    if not roll_correct:
                        if display_roll < target_roll_min_step_2:
                            error_messages.append("अपने सिर को और बाईं ओर झुकाएं।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपने सिर को और बाईं ओर झुकाएं।")
                                last_speech_time = current_time
                        elif display_roll > target_roll_max_step_2:
                            error_messages.append("अपने सिर को दाईं ओर झुकाएं।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपने सिर को दाईं ओर झुकाएं।")
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
                        speak("इस स्थिति को 45 सेकंड तक रखें।")
                        last_speech_time = current_time
                    if bppv_start_time is None:
                        bppv_start_time = current_time
                    bppv_pose_held_time = current_time - bppv_start_time
                    remaining_time = max(0, bppv_duration_threshold - bppv_pose_held_time)
                    cv2.putText(frame, f"Hold Head at this position for {remaining_time:.1f}s",
                                (frame_width // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if current_time - last_speech_time > speech_interval:
                        speak(f"अपने सिर को इस स्थिति में {remaining_time:.1f} सेकंड तक रखें।")
                        last_speech_time = current_time
                    if bppv_pose_held_time >= bppv_duration_threshold:
                        bppv_step_4 = True
                        correct_beep.play()
                        speak("तीसरा चरण पूरा। अपने सिर को 2 से 3 बार हिलाएं और फिर बिस्तर के विपरीत तरफ साधारण स्थिति में 5 सेकंड के लिए बैठें।", critical=True)
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
                            error_messages.append("अपने सिर को और बाईं ओर मोड़ें।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपने सिर को और बाईं ओर मोड़ें।")
                                last_speech_time = current_time
                        elif display_yaw > target_yaw_max_step_3:
                            error_messages.append("अपने सिर को दाईं ओर वापस मोड़ें।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपने सिर को दाईं ओर वापस मोड़ें।")
                                last_speech_time = current_time
                    if not pitch_correct:
                        if display_pitch < target_pitch_min_step_3:
                            error_messages.append("अपने सिर को और ऊपर झुकाएं।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपने सिर को और ऊपर झुकाएं।")
                                last_speech_time = current_time
                        elif display_pitch > target_pitch_max_step_3:
                            error_messages.append("अपने सिर को नीचे झुकाएं।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपने सिर को नीचे झुकाएं।")
                                last_speech_time = current_time
                    if not roll_correct:
                        if display_roll < target_roll_min_step_3:
                            error_messages.append("अपने सिर को और दाईं ओर झुकाएं।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपने सिर को और दाईं ओर झुकाएं।")
                                last_speech_time = current_time
                        elif display_roll > target_roll_max_step_3:
                            error_messages.append("अपने सिर को बाईं ओर झुकाएं।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपने सिर को बाईं ओर झुकाएं।")
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
                        speak("इस स्थिति को 5 सेकंड तक रखें।")
                        last_speech_time = current_time
                        in_correct_pose_step_4 = True
                    if bppv_start_time is None:
                        bppv_start_time = current_time
                    bppv_pose_held_time = current_time - bppv_start_time
                    remaining_time = max(0, neutral_hold_threshold - bppv_pose_held_time)
                    cv2.putText(frame, f"Hold Neutral Position for {remaining_time:.1f}s",
                                (frame_width // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if current_time - last_speech_time > speech_interval:
                        speak(f"अपने सिर को इस स्थिति में {remaining_time:.1f} सेकंड तक रखें।")
                        last_speech_time = current_time
                    if bppv_pose_held_time >= neutral_hold_threshold:
                        all_missions_complete = True
                        print("BPPV Step 4 Complete!")
                        speak("आपने इसे सफलतापूर्वक पूरा कर लिया है।", critical=True)
                        last_speech_time = current_time
                else:
                    bppv_start_time = None
                    in_correct_pose_step_4 = False
                    error_messages = []
                    if not yaw_correct:
                        if display_yaw > target_yaw_min_step_4:
                            error_messages.append("अपने सिर को और दाईं ओर मोड़ें।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपने सिर को और दाईं ओर मोड़ें।")
                                last_speech_time = current_time
                        elif display_yaw < target_yaw_max_step_4:
                            error_messages.append("अपने सिर को और बाईं ओर मोड़ें।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपने सिर को और बाईं ओर मोड़ें।")
                                last_speech_time = current_time
                    if not pitch_correct:
                        if display_pitch < target_pitch_min_step_4:
                            error_messages.append("अपना सिर ऊपर उठाएं।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपना सिर ऊपर उठाएं।")
                                last_speech_time = current_time
                        elif display_pitch > target_pitch_max_step_4:
                            error_messages.append("अपना सिर नीचे करें।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपना सिर नीचे करें।")
                                last_speech_time = current_time
                    if not roll_correct:
                        if display_roll > target_roll_min_step_4:
                            error_messages.append("अपने सिर को बाईं ओर झुकाएं।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपने सिर को बाईं ओर झुकाएं।")
                                last_speech_time = current_time
                        elif display_roll < target_roll_max_step_4:
                            error_messages.append("अपने सिर को दाईं ओर झुकाएं।")
                            if current_time - last_speech_time > speech_interval:
                                speak("अपने सिर को दाईं ओर झुकाएं।")
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

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    pygame.mixer.quit()
    critical_speech_queue.put(None)
    noncritical_speech_queue.put(None)
    speech_thread.join(timeout=1)

if __name__ == "__main__":
    run()