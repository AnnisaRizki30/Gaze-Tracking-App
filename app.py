from flask import Flask, render_template, jsonify, Response
from flask_cors import CORS  # Import CORS
import cv2
from gaze_tracking import GazeTracking

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

@app.route('/')
def index():
    # Render the HTML template
    return render_template('index.html')

@app.route('/face_tracking', methods=['GET'])
def face_tracking():
    # Capture a frame from the webcam
    success, frame = webcam.read()
    if not success:
        return jsonify({"error": "Failed to capture frame from webcam"}), 500

    # Analyze the frame using GazeTracking
    gaze.refresh(frame)

    # Get pupil coordinates and convert to serializable types
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    data = {
        "status": "",
        "left_pupil": [int(coord) for coord in left_pupil] if left_pupil else None,
        "right_pupil": [int(coord) for coord in right_pupil] if right_pupil else None
    }

    if gaze.is_blinking():
        data["status"] = "Blinking"
    elif gaze.is_right():
        data["status"] = "Looking right"
    elif gaze.is_left():
        data["status"] = "Looking left"
    elif gaze.is_center():
        data["status"] = "Looking center"

    # Return the data as a JSON response
    return jsonify(data)


@app.route('/video_feed', methods=['GET'])
def video_feed():
    def generate_frames():
        while True:
            success, frame = webcam.read()
            if not success:
                break

            # Analyze the frame and annotate it
            gaze.refresh(frame)
            frame = gaze.annotated_frame()

            # Encode the frame as a JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()

            # Yield the frame as part of the multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
