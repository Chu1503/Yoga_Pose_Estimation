from flask import Flask, render_template, request, Response
import cv2
import mediapipe as mp
import numpy as np
import time
import base64
from threading import Thread

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def generate_pose_frames(pose_name):
    global timer_running
    global total_elapsed_time

    timer_running = False
    total_elapsed_time = 0

    prev_time = 0
    fps = 0
    timer_start_time = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                # Get coordinates
                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y] 
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y] 

                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # Calculate angle
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                # Visualize angle
                cv2.putText(image, f'{left_elbow_angle:.2f}', 
                            tuple(np.multiply(left_elbow, [1280, 720]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, f'{right_elbow_angle:.2f}', 
                            tuple(np.multiply(right_elbow, [1280, 720]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, f'{left_knee_angle:.2f}', 
                            tuple(np.multiply(left_knee, [1280, 720]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, f'{right_knee_angle:.2f}', 
                            tuple(np.multiply(right_knee, [1280, 720]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,255,255), 2, cv2.LINE_AA)

                # Encode image to base64
                _, img_encoded = cv2.imencode('.jpg', image)
                img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

            except:
                pass

            # Check the conditions for starting/pausing the timer
            if (160 < left_elbow_angle < 180 and 
                160 < right_elbow_angle <180 and 
                165 < left_knee_angle < 185 and 
                165 < right_knee_angle < 185):
                if not timer_running:
                    timer_start_time = time.time() - total_elapsed_time
                    timer_running = True
                        
            else:
                if timer_running:
                    total_elapsed_time = time.time() - timer_start_time
                    timer_running = False
                        
            # Display the timer
            if timer_running:
                timer_display = f'Timer: {int(total_elapsed_time)}s'
            else:
                timer_display = f'Timer: Paused'
                    
            cv2.putText(image, timer_display, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Render detections
            
            if (160 < left_elbow_angle < 180 and 
                160 < right_elbow_angle <180 and 
                150 < left_knee_angle < 180 and 
                110 < right_knee_angle < 140):
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                                    )
            else:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2) 
                                    )
                
            # Calculate and display FPS
            current_time = time.time()
            elapsed_time = current_time - prev_time
            fps = 1 / elapsed_time
            prev_time = current_time
            
            cv2.putText(image, f'FPS: {int(fps)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Encode image to base64
            _, img_encoded = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_base64.encode() + b'\r\n')

        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_pose_detection', methods=['POST'])
def start_pose_detection():
    chosen_pose = request.form['pose']
    if chosen_pose == "Warrior Pose":
        return Response(generate_pose_frames(chosen_pose), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid Pose"

if __name__ == '__main__':
    app.run(debug=True)
