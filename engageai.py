import dotenv

dotenv.load_dotenv()

import cv2
from collections import Counter, deque
from deepface import DeepFace
import numpy as np
import openai
import os

# Set up OpenAI API key
from openai import OpenAI

# Define the emotions associated with engagement and disengagement
engaged_emotions = ['neutral', 'happy', 'surprise']
disengaged_emotions = ['sad', 'bored', 'disgust', 'angry']

# Rolling window size for smoothing engagement detection
rolling_window_size = 5

# Load pre-trained face and eye detection models (Haar cascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def generate_advice(detected_emotion):
    if detected_emotion in disengaged_emotions:
        prompt = (
            f"It seems the user is showing signs of {detected_emotion}. "
            "What 1 specific advice would you give to re-engage them? e.g. ask a question, provide a challenge, etc."
        )
    else:
        prompt = (
            "The user appears engaged. How can we maintain their interest?"
        )
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
)
    # Using the updated chat completions API
    chat_completion = client.chat.completions.create(
        model="gpt-4o",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are an AI assistant that provides helpful advice for students/learners engagement."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=50,
    )
    
    advice = chat_completion.choices[0].message.content
    return advice

def is_looking_at_camera(eyes):
    if len(eyes) != 2:
        return False  # We need exactly two eyes to make a reliable estimate

    left_eye = eyes[0]
    right_eye = eyes[1]

    left_eye_center = np.array([left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2])
    right_eye_center = np.array([right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2])

    horizontal_distance = abs(left_eye_center[0] - right_eye_center[0])
    vertical_distance = abs(left_eye_center[1] - right_eye_center[1])

    return vertical_distance < horizontal_distance * 0.3  # Adjust the threshold as needed

def classify_engagement(face_info, eyes):
    dominant_emotion = face_info['dominant_emotion']
    looking_at_camera = is_looking_at_camera(eyes)
    if looking_at_camera:
        if dominant_emotion in engaged_emotions:
            return "Engaged", "Looking at Camera"
        else:
            return "Uncertain", "Looking at Camera"
    else:
        return "Disengaged", "Not Looking at Camera"

def display_info_on_frame(frame, engagement, looking_status, face_position=None, global_status=False):
    if global_status:
        # Display global engagement and gaze status at the top center of the frame
        engagement_text = f"Global Engagement: {engagement}"
        looking_text = f"Global Gaze: {looking_status}"
        font_scale = 1.0
        font_thickness = 2
        font_color = (0, 255, 0)  # Green color for visibility

        text_size_eng = cv2.getTextSize(engagement_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x_eng = int((frame.shape[1] - text_size_eng[0]) / 2)
        text_y_eng = text_size_eng[1] + 10
        cv2.putText(frame, engagement_text, (text_x_eng, text_y_eng), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

        text_size_look = cv2.getTextSize(looking_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x_look = int((frame.shape[1] - text_size_look[0]) / 2)
        text_y_look = text_y_eng + text_size_look[1] + 10
        cv2.putText(frame, looking_text, (text_x_look, text_y_look), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
    
    else:
        # Display individual engagement and gaze status near the face
        engagement_text = f"Engagement: {engagement}"
        looking_text = f"Gaze: {looking_status}"
        font_scale = 0.6
        font_thickness = 2
        font_color = (255, 0, 0)  # Blue color for visibility

        x, y, w, h = face_position

        cv2.putText(frame, engagement_text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
        cv2.putText(frame, looking_text, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

    # # Display advice in the lower right corner of the frame
    # if advice:
    #     advice_font_scale = 0.5
    #     advice_thickness = 1
    #     advice_color = (255, 255, 255)  # White color for visibility
    #     text_size_adv = cv2.getTextSize(advice, cv2.FONT_HERSHEY_SIMPLEX, advice_font_scale, advice_thickness)[0]
    #     text_x_adv = frame.shape[1] - text_size_adv[0] - 10
    #     text_y_adv = frame.shape[0] - 10
    #     cv2.putText(frame, advice, (text_x_adv, text_y_adv), cv2.FONT_HERSHEY_SIMPLEX, advice_font_scale, advice_color, advice_thickness)

def analyze_and_display(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    individual_engagement_statuses = []
    individual_looking_statuses = []

    advice = ""

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi)
        
        engagement_window = deque(maxlen=rolling_window_size)
        looking_status_window = deque(maxlen=rolling_window_size)

        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            if isinstance(result, list):
                for face in result:
                    engagement, looking_status = classify_engagement(face, eyes)
                    engagement_window.append(engagement)
                    looking_status_window.append(looking_status)
            else:
                engagement, looking_status = classify_engagement(result, eyes)
                engagement_window.append(engagement)
                looking_status_window.append(looking_status)

        except Exception as e:
            print(e)

        # Get the most common engagement and looking status for this face
        smoothed_engagement = Counter(engagement_window).most_common(1)[0][0] if engagement_window else "Unknown"
        smoothed_looking = Counter(looking_status_window).most_common(1)[0][0] if looking_status_window else "Unknown"

        # Collect individual statuses for global calculation
        individual_engagement_statuses.append(smoothed_engagement)
        individual_looking_statuses.append(smoothed_looking)

        # Display the information for this face
        display_info_on_frame(frame, smoothed_engagement, smoothed_looking, (x, y, w, h))

    # Calculate and display global status if any faces are detected
    if individual_engagement_statuses and individual_looking_statuses:
        global_engagement = Counter(individual_engagement_statuses).most_common(1)[0][0]
        global_looking = Counter(individual_looking_statuses).most_common(1)[0][0]
        #advice = generate_advice(global_engagement)
        display_info_on_frame(frame, global_engagement, global_looking, global_status=True)

    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream from camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = analyze_and_display(frame)

        cv2.imshow('Engagement and Gaze Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
