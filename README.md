# AI-Based BPPV Maneuver Guider (Gans Repositioning Maneuver)

## Overview

This project is an AI-powered application designed to guide users through the **Gans Repositioning Maneuver**, a treatment for Benign Paroxysmal Positional Vertigo (BPPV). It uses computer vision to track head position in real-time and provides step-by-step audio and visual feedback to ensure the maneuver is performed correctly.

## Features

-   **Real-time Pose Estimation**: Uses MediaPipe Pose to track head orientation (Yaw, Pitch, Roll) with high precision.
-   **Step-by-Step Guidance**: Automatically detects when a step is completed and guides the user to the next stage of the maneuver.
-   **Multi-Language Support**: Full voice and text instructions in **English** and **Hindi**.
-   **Bilateral Support**: Specialized guides for both **Left** and **Right** affected ears.
-   **Audio Feedback**:
    -   **Text-to-Speech (TTS)**: Verbal instructions for each step and corrections if the user drifts from the target pose.
    -   **Sound Effects**: Distinct beeps for correct (green) and incorrect (red) poses.
-   **Visual Feedback**:
    -   On-screen bounding box turns **Green** when the pose is correct and **Red** when adjustment is needed.
    -   Real-time display of current head angles (Yaw, Pitch, Roll) and countdown timers.
-   **Kalman Filter Smoothing**: Implements Kalman filtering to ensure stable and smooth angle detection, reducing jitter.

## Prerequisites

-   **Python 3.7+**
-   **Webcam**: A functional webcam is required for pose detection.
-   **Audio Output**: Speakers or headphones for voice instructions.

## Installation

1.  **Clone or Download** the repository to your local machine.
2.  **Navigate** to the project directory in your terminal.
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Key dependencies include: `opencv-python`, `mediapipe`, `numpy`, `pyttsx3`, `pygame`, `filterpy`.*

### Hindi Voice Setup (Optional - For Hindi Instructions)

To enable the "Microsoft Hemant" Hindi voice for TTS:

1.  **Check for Voice**: Go to **Settings > Time & Language > Speech** on Windows and check if "Hindi (India)" is installed.
2.  **Install Registry Key**: If the voice is not detected by Python, double-click the provided `hemant.reg` file to register the voice.
3.  **Restart**: Restart your computer after applying the registry change.

## Usage

1.  **Run the Main Application**:
    ```bash
    python main.py
    ```

2.  **Select Affected Ear**:
    -   Enter `Left` or `Right` when prompted.

3.  **Select Language**:
    -   Enter `English` or `Hindi` when prompted.

4.  **Follow the Instructions**:
    -   **Visibility Check**: Ensure your full body is visible in the camera frame.
    -   **Calibration**: Align your head within the on-screen box until it turns green.
    -   **Maneuver Steps**: Follow the voice commands to turn your head, lie down, and roll as instructed. The system will count down (e.g., 45 seconds) for each holding position.

## Project Structure

-   `main.py`: Entry point of the application. Handles user input for ear and language selection.
-   `left_ear_english.py` / `right_ear_english.py`: Logic for Left/Right ear maneuvers with English instructions.
-   `left_ear_hindi.py` / `right_ear_hindi.py`: Logic for Left/Right ear maneuvers with Hindi instructions.
-   `requirements.txt`: List of Python dependencies.
-   `read.txt`: Detailed setup notes and troubleshooting for Hindi voice support.
-   `hemant.reg`: Registry file for enabling the specific Hindi TTS voice.
-   `correct.wav` / `incorrect.wav`: Audio assets for feedback.

## Troubleshooting

-   **Camera not opening**: Ensure no other application is using the webcam.
-   **Voice not working**: Verify your system volume and that `pyttsx3` is installed correctly. For Hindi, follow the specific setup steps in `read.txt`.
-   **Detection issues**: Ensure good lighting and that your full upper body is visible in the frame.

