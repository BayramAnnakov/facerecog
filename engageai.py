import cv2
import dlib
from collections import Counter, deque
from deepface import DeepFace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import logging

import os

shape_predictor_path = "shape_predictor_68_face_landmarks.dat"

if not os.path.isfile(shape_predictor_path):
    raise ValueError(f"The shape predictor file at {shape_predictor_path} was not found. Make sure the path is correct.")

# Load dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the emotions associated with engagement and disengagement
engaged_emotions = {'neutral', 'happy', 'surprise'}
disengaged_emotions = {'sad', 'bored', 'disgust', 'angry'}

# Rolling window size for smoothing engagement detection
rolling_window_size = 5

# Initialize engagement history for graph
engagement_history = deque(maxlen=100)  # Keep the last 100 engagement levels
smoothed_engagement_history = deque(maxlen=100)  # Keep the last 100 smoothed engagement levels



def ensure_display_status(status, default="Unknown"):
    return status if status else default

def is_looking_at_camera(landmarks):
    # Landmark points for eyes
    left_eye_points = list(range(36, 42))
    right_eye_points = list(range(42, 48))

    # Calculate the centers of the eyes
    left_eye_center = np.mean([(landmarks.part(p).x, landmarks.part(p).y) for p in left_eye_points], axis=0)
    right_eye_center = np.mean([(landmarks.part(p).x, landmarks.part(p).y) for p in right_eye_points], axis=0)

    # Calculate the distance between the eyes
    eye_distance = np.linalg.norm(right_eye_center - left_eye_center)

    # Measure the vertical distance from the eyes to the midpoint between them
    midpoint = (left_eye_center + right_eye_center) / 2
    vertical_dist_left = abs(left_eye_center[1] - midpoint[1])
    vertical_dist_right = abs(right_eye_center[1] - midpoint[1])

    # If the vertical distances are similar, the person is more likely looking at the camera
    if vertical_dist_left < eye_distance * 0.2 and vertical_dist_right < eye_distance * 0.2:
        return True, "Looking at Camera"
    else:
        return False, "Not Looking at Camera"

def classify_engagement(face_info, landmarks):
    dominant_emotion = face_info.get('dominant_emotion', 'neutral')
    looking_at_camera, gaze_status = is_looking_at_camera(landmarks)

    if looking_at_camera:
        if dominant_emotion in engaged_emotions:
            return "Engaged", gaze_status
        else:
            return "Uncertain", gaze_status
    else:
        return "Disengaged", gaze_status

def display_info_on_frame(frame, engagement, looking_status, face_position=None, global_status=False):
    engagement = ensure_display_status(engagement, "Unknown")
    looking_status = ensure_display_status(looking_status, "Unknown")

    font_scale = 1.0 if global_status else 0.6
    font_thickness = 2
    font_color = (0, 255, 0) if global_status else (255, 0, 0)

    if global_status:
        engagement_text = f"Global Engagement: {engagement}"
        looking_text = f"Global Gaze: {looking_status}"
        display_text_on_frame(frame, engagement_text, looking_text, font_scale, font_thickness, font_color)
    else:
        if face_position:
            x, y, w, h = face_position
            engagement_text = f"Engagement: {engagement}"
            looking_text = f"Gaze: {looking_status}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), font_color, 2)
            display_text_on_frame(frame, engagement_text, looking_text, font_scale, font_thickness, font_color, x, y - 40)

def display_text_on_frame(frame, engagement_text, looking_text, font_scale, font_thickness, font_color, x=None, y=None):
    try:
        text_size_eng = cv2.getTextSize(engagement_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_size_look = cv2.getTextSize(looking_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

        if x is None and y is None:
            x = int((frame.shape[1] - text_size_eng[0]) / 2)
            y = text_size_eng[1] + 10

        cv2.putText(frame, engagement_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
        cv2.putText(frame, looking_text, (x, y + text_size_look[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

    except Exception as e:
        logging.error(f"Error displaying text on frame: {e}")

def analyze_and_display(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    individual_engagement_statuses = []
    individual_looking_statuses = []

    for face in faces:
        landmarks = predictor(gray, face)
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_roi = frame[y:y+h, x:x+w]

        # Analyze the face using DeepFace for emotion detection
        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            if isinstance(result, list):
                for face in result:
                    engagement, looking_status = classify_engagement(face, landmarks)
                    individual_engagement_statuses.append(engagement)
                    individual_looking_statuses.append(looking_status)
            else:
                engagement, looking_status = classify_engagement(result, landmarks)
                individual_engagement_statuses.append(engagement)
                individual_looking_statuses.append(looking_status)

        except Exception as e:
            logging.error(f"Error during DeepFace analysis: {e}")
            engagement, looking_status = "Unknown", "Unknown"

        display_info_on_frame(frame, engagement, looking_status, (x, y, w, h))

    if individual_engagement_statuses and individual_looking_statuses:
        global_engagement = Counter(individual_engagement_statuses).most_common(1)[0][0]
        global_looking = Counter(individual_looking_statuses).most_common(1)[0][0]
        display_info_on_frame(frame, global_engagement, global_looking, global_status=True)

        engagement_numeric = 1 if global_engagement == "Engaged" else 0
        engagement_history.append(engagement_numeric)

        smoothed_value = sum(engagement_history) / len(engagement_history)
        smoothed_engagement_history.append(smoothed_value)

    return frame

def draw_graph_on_frame(frame):
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(smoothed_engagement_history, label='Smoothed Engagement Level')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0, len(smoothed_engagement_history))
    ax.set_title('Smoothed Engagement Level Dynamics')
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Smoothed Engagement Level')
    ax.legend(loc='upper right')
    fig.tight_layout()

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    graph_image = np.asarray(buf, dtype=np.uint8)

    graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGBA2BGR)
    graph_width = frame.shape[1] // 3
    graph_height = int(graph_width / 1.5)
    graph_image = cv2.resize(graph_image, (graph_width, graph_height))

    x_offset = frame.shape[1] - graph_width - 10
    y_offset = frame.shape[0] - graph_height - 10
    frame[y_offset:y_offset+graph_height, x_offset:x_offset+graph_width] = graph_image

    plt.close(fig)
    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open video stream from camera.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture frame.")
                break

            frame = analyze_and_display(frame)
            frame = draw_graph_on_frame(frame)

            cv2.imshow('Engagement and Gaze Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
