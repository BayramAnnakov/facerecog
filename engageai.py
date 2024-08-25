import cv2
from collections import Counter, deque
from deepface import DeepFace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg

# Define the emotions associated with engagement and disengagement
engaged_emotions = ['neutral', 'happy', 'surprise']
disengaged_emotions = ['sad', 'bored', 'disgust', 'angry']

# Rolling window size for smoothing engagement detection
rolling_window_size = 5

# Initialize engagement history for graph
engagement_history = deque(maxlen=100)  # Keep the last 100 engagement levels

# Initialize smoothed engagement history
smoothed_engagement_history = deque(maxlen=100)  # Keep the last 100 smoothed engagement levels

# Load pre-trained face and eye detection models (Haar cascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def ensure_display_status(status, default="Unknown"):
    return status if status else default

def is_looking_at_camera(eyes):
    if len(eyes) != 2:
        return False, "Not Looking at Camera"  # We need exactly two eyes to make a reliable estimate

    # Sort eyes by their x-coordinate (left to right)
    eyes = sorted(eyes, key=lambda eye: eye[0])

    left_eye = eyes[0]
    right_eye = eyes[1]

    # Calculate the center of each eye
    left_eye_center = np.array([left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2])
    right_eye_center = np.array([right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2])

    # Calculate the distance between the eyes
    eye_distance = np.linalg.norm(right_eye_center - left_eye_center)

    # Calculate the midpoint between the two eyes
    midpoint = (left_eye_center + right_eye_center) / 2

    # Measure the vertical distance from the eyes to the midpoint
    vertical_dist_left = abs(left_eye_center[1] - midpoint[1])
    vertical_dist_right = abs(right_eye_center[1] - midpoint[1])

    # If the vertical distances are similar, the person is more likely looking at the camera
    # Adjust the sensitivity by changing the multiplier
    if vertical_dist_left < eye_distance * 0.2 and vertical_dist_right < eye_distance * 0.2:
        return True, "Looking at Camera"
    else:
        return False, "Not Looking at Camera"

def classify_engagement(face_info, eyes):
    dominant_emotion = face_info.get('dominant_emotion', 'neutral')  # Default to 'neutral' if key not found
    looking_at_camera, gaze_status = is_looking_at_camera(eyes)
    
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

    try:
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

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), font_color, 2)

            # Display the status text above the rectangle
            cv2.putText(frame, engagement_text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
            cv2.putText(frame, looking_text, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
    
    except Exception as e:
        print(f"Error displaying text on frame: {e}")

def analyze_and_display(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    individual_engagement_statuses = []
    individual_looking_statuses = []

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi)

        # Analyze the face using DeepFace for emotion detection
        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            if isinstance(result, list):
                for face in result:
                    engagement, looking_status = classify_engagement(face, eyes)
                    individual_engagement_statuses.append(engagement)
                    individual_looking_statuses.append(looking_status)
            else:
                engagement, looking_status = classify_engagement(result, eyes)
                individual_engagement_statuses.append(engagement)
                individual_looking_statuses.append(looking_status)

        except Exception as e:
            print(f"Error during DeepFace analysis: {e}")

        # Display the information for this face
        display_info_on_frame(frame, engagement, looking_status, (x, y, w, h))

    # Calculate and display global status if any faces are detected
    if individual_engagement_statuses and individual_looking_statuses:
        global_engagement = Counter(individual_engagement_statuses).most_common(1)[0][0]
        global_looking = Counter(individual_looking_statuses).most_common(1)[0][0]
        display_info_on_frame(frame, global_engagement, global_looking, global_status=True)

        # Convert engagement status to numeric for plotting and smoothing
        engagement_numeric = 1 if global_engagement == "Engaged" else 0
        engagement_history.append(engagement_numeric)

        # Calculate a simple moving average for smoothing
        if len(engagement_history) > 1:
            smoothed_value = sum(engagement_history) / len(engagement_history)
        else:
            smoothed_value = engagement_numeric
        
        smoothed_engagement_history.append(smoothed_value)

    return frame

def draw_graph_on_frame(frame):
    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(smoothed_engagement_history, label='Smoothed Engagement Level')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0, len(smoothed_engagement_history))  # Adjust X-axis to fit the graph data
    ax.set_title('Smoothed Engagement Level Dynamics')
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Smoothed Engagement Level')
    ax.legend(loc='upper right')
    fig.tight_layout()  # Ensure no clipping

    # Render the graph to an image
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    graph_image = np.asarray(buf, dtype=np.uint8)

    # Resize the graph to fit on the video frame
    graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGBA2BGR)
    graph_width = frame.shape[1] // 3  # Make the graph one-third of the frame's width
    graph_height = int(graph_width / 1.5)  # Maintain the aspect ratio
    graph_image = cv2.resize(graph_image, (graph_width, graph_height))

    # Overlay the graph on the bottom right of the video frame
    x_offset = frame.shape[1] - graph_width - 10  # 10-pixel margin from the right edge
    y_offset = frame.shape[0] - graph_height - 10  # 10-pixel margin from the bottom edge
    frame[y_offset:y_offset+graph_height, x_offset:x_offset+graph_width] = graph_image

    # Close the figure to free memory
    plt.close(fig)

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

        # Draw the graph on the frame
        frame = draw_graph_on_frame(frame)

        cv2.imshow('Engagement and Gaze Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
