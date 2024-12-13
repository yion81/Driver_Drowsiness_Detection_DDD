from flask import Flask, Response, render_template, jsonify
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model


app = Flask(__name__)

is_drowsy = False

model = load_model(r'C:\Codes\AI Model\Driver Drowsiness V2\DDD_model_50epochs.h5')

mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# cam
cap = cv2.VideoCapture(0)

# eye landmarks
all_left_eye_idxs = list(mp_facemesh.FACEMESH_LEFT_EYE)
all_right_eye_idxs = list(mp_facemesh.FACEMESH_RIGHT_EYE)
all_idxs = set(np.ravel(all_left_eye_idxs)).union(set(np.ravel(all_right_eye_idxs)))
chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs
FACEMESH_FACE = list(range(0, 478))
IMG_SIZE = 145

def draw_landmarks(img_dt, face_landmarks):
    image_drawing_tool = img_dt.copy()
    connections_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color=(255, 255, 255))
    mp_drawing.draw_landmarks(
        image=image_drawing_tool,
        landmark_list=face_landmarks,
        connections=mp_facemesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=connections_drawing_spec,
    )

    landmarks = face_landmarks.landmark
    imgH, imgW, _ = img_dt.shape

    for landmark_idx, landmark in enumerate(landmarks):
        if landmark_idx in all_idxs:
            pred_cord = denormalize_coordinates(landmark.x, landmark.y, imgW, imgH)
            cv2.circle(image_drawing_tool, pred_cord, 3, (255, 255, 255), -1)
        if landmark_idx in all_chosen_idxs:
            pred_cord = denormalize_coordinates(landmark.x, landmark.y, imgW, imgH)
            cv2.circle(image_drawing_tool, pred_cord, 3, (255, 255, 255), -1)

    return image_drawing_tool

def preprocess_frame(frame):
    """Preprocess the frame for the model."""
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC) # resize and sharpening
    frame = frame.astype('float32') / 255.0  # normalize
    frame = np.expand_dims(frame, axis=0)  # reshape
    return frame

def crop_face_region(frame, landmarks, padding=0.1):
    """Crop the entire face region based on face landmarks."""
    imgH, imgW, _ = frame.shape
    points = []

    for idx in FACEMESH_FACE:
        if idx < len(landmarks): 
            normalized_landmark = landmarks[idx]
            x = int(normalized_landmark.x * imgW)
            y = int(normalized_landmark.y * imgH)
            if 0 <= x < imgW and 0 <= y < imgH:
                points.append((x, y))

    if not points:
        raise ValueError("No valid points found for cropping the face region.")

    # box
    x_coords, y_coords = zip(*points)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    x_padding = int(padding * (x_max - x_min))
    y_padding = int(padding * (y_max - y_min))
    x_min = max(0, x_min - x_padding)
    y_min = max(0, y_min - y_padding)
    x_max = min(imgW, x_max + x_padding)
    y_max = min(imgH, y_max + y_padding)

    # crop
    cropped_frame = frame[y_min:y_max, x_min:x_max]
    return cropped_frame

# intializes
@app.route('/')
def index():
    return render_template('index.html')

# when u press logo
@app.route('/index_view')
def index_view():
    return render_template('index.html')

@app.route('/standard_view')
def standard_view():
    return render_template('standard_view.html')

@app.route('/standard_ai_view')
def standard_ai_view():
    return render_template('ai_view.html')



# bugged. see javascript or maybe its this code, or maybe the generate_frame()
@app.route('/get_drowsy_status')
def get_drowsy_status():
    print(f"Returning drowsy status: {is_drowsy}")
    return jsonify({'is_drowsy': bool(is_drowsy)})

# i want to kill myself this is 4 hours of debugging just for typecasting is_drowsy into bool

@app.route('/video_feed_std')
def video_feed_std():
    global is_drowsy
    
    face_mesh = mp_facemesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def generate_frames():
        global is_drowsy
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to capture image")
                break

            frame = cv2.flip(frame, 1) # flip it
            results = face_mesh.process(frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    frame = draw_landmarks(frame, face_landmarks)

                    cropped_frame = crop_face_region(frame, face_landmarks.landmark)

                    input_frame = preprocess_frame(cropped_frame)

                    prediction = model.predict(input_frame)

                    # print(f"Before prediction: {is_drowsy}")
                    is_drowsy = prediction[0][0] < 0.7
                    # print(f"After prediction: {is_drowsy}")
                    if is_drowsy:
                        prediction_label = "Drowsy"
                        prediction_value = prediction[0][0]  # confidence sc drowsy
                    else:
                        prediction_label = "Awake"
                        prediction_value = prediction[0][0]  # confidence sc awake

                    # Annotate the original frame
                    cv2.putText(frame, f"{prediction_label} ({prediction_value:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    print(f"Prediction: {prediction_label}, Predict Value: {prediction_value:.2f}")

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode image")
                break
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_std_ai')
def video_feed_std_ai():
    global is_drowsy
    face_mesh = mp_facemesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def generate_frames():
        global is_drowsy
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to capture image")
                break

            frame = cv2.flip(frame, 1)
            results = face_mesh.process(frame)

            preprocessed_display = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    frame = draw_landmarks(frame, face_landmarks)

                    cropped_frame = crop_face_region(frame, face_landmarks.landmark)

                    input_frame = preprocess_frame(cropped_frame)

                    prediction = model.predict(input_frame)

                    is_drowsy = prediction[0][0] < 0.7
                    if is_drowsy:
                        prediction_label = "Drowsy"
                        prediction_value = prediction[0][0]  # confidence sc drowsy
                    else:
                        prediction_label = "Awake"
                        prediction_value = prediction[0][0]  # confidence sc awake

                    # Annotate the original frame
                    cv2.putText(frame, f"{prediction_label} ({prediction_value:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    print(f"Prediction: {prediction_label}, Predict Value: {prediction_value:.2f}")

                    preprocessed_display = (input_frame[0] * 255).astype('uint8')  # denormalize
                    if preprocessed_display.shape[-1] == 1:
                        preprocessed_display = cv2.cvtColor(preprocessed_display, cv2.COLOR_GRAY2BGR)

            # resize the ai frame to be the std's one
            preprocessed_display_resized = cv2.resize(preprocessed_display, (frame.shape[1], frame.shape[0]))

            # combine both frame std and ai
            combined_frame = np.hstack((frame, preprocessed_display_resized))

            ret, buffer = cv2.imencode('.jpg', combined_frame)
            if not ret:
                print("Failed to encode image")
                break
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

try:
    cap.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(f"Error releasing camera: {e}")
