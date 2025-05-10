import face_recognition
import cv2
import numpy as np
import time
import pickle
import os
import sys
from picamera2 import Picamera2
from gpiozero import LED
from luma.oled.device import ssd1306
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from PIL import ImageFont, ImageDraw

# Set this to a higher value (e.g., 4 or 5) for better performance, or lower (e.g., 2) for better accuracy.
cv_scaler = 4  # 1 = full resolution, 2 = half, 4 = quarter, etc.

def log_message(message):
    try:
        with open("log.txt", "a") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        print(message)  # Also print to console
    except Exception as e:
        print(f"Error writing to log: {e}", file=sys.stderr)

# Check if display is available
try:
    display = os.environ.get('DISPLAY')
    if display:
        log_message(f"Display found: {display}")
        has_display = True
    else:
        log_message("No display found, running in headless mode")
        has_display = False
except Exception as e:
    log_message(f"Error checking display: {e}")
    has_display = False

# Initialize GPIO pins
try:
    RED_LED = LED(17, initial_value=False)  # GPIO 17 for red LED
    GREEN_LED = LED(27, initial_value=False)  # GPIO 27 for green LED
    log_message("GPIO pins initialized successfully")
except Exception as e:
    log_message(f"Error initializing GPIO pins: {e}")
    sys.exit(1)

# Initialize OLED display
try:
    serial = i2c(port=1, address=0x3C)
    device = ssd1306(serial)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    log_message("OLED display initialized successfully")
except Exception as e:
    log_message(f"Error initializing OLED display: {e}")
    # Continue without OLED display
    device = None
    font = None

# Load pre-trained face encodings
known_face_encodings = []
known_face_names = []
try:
    if os.path.exists("encodings.pickle"):
        with open("encodings.pickle", "rb") as f:
            data = pickle.loads(f.read())
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
            log_message(f"Loaded {len(known_face_names)} known faces")
    else:
        log_message("Warning: encodings.pickle not found. Starting with no known faces.")
except Exception as e:
    log_message(f"Error loading encodings: {e}")
    # Continue with empty face encodings
    known_face_encodings = []
    known_face_names = []

# Initialize the camera (Pi 5 specific configuration)
try:
    picam2 = Picamera2()
    # Configure for better performance on Pi 5
    config = picam2.create_preview_configuration(
        main={"format": 'XRGB8888', "size": (640, 480)},
        controls={"FrameDurationLimits": (33333, 33333)}  # 30fps
    )
    picam2.configure(config)
    picam2.start()
    log_message("Camera initialized successfully")
except Exception as e:
    log_message(f"Error initializing camera: {e}")
    sys.exit(1)

# Initialize our variables
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

def update_display(message, authorized=False):
    """Update the OLED display with the given message. Always clear the display first."""
    if device is None:
        return
    try:
        with canvas(device) as draw:
            draw.rectangle(device.bounding_box, outline="white", fill="black")
            draw.text((10, 10), message, fill="white", font=font)
            if authorized:
                draw.text((10, 30), "Access Granted", fill="white", font=font)
            elif message.startswith("Welcome "):
                pass
            elif not authorized and not message.startswith("Welcome"):
                draw.text((10, 30), "Access Denied", fill="white", font=font)
    except Exception as e:
        log_message(f"Error updating display: {e}")

def process_frame(frame, state):
    global face_locations, face_encodings, face_names
    try:
        if frame is None:
            log_message("Received empty frame from camera")
            return None

        # Resize the frame for better performance
        resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
        rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Find all faces in the frame
        face_locations = face_recognition.face_locations(rgb_resized_frame)
        face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
        
        face_names = []
        authorized_face_detected = False
        authorized_name = None
        
        if known_face_encodings:  # Only try to match if we have known faces
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    authorized_face_detected = True
                    authorized_name = name
                face_names.append(name)
        else:
            face_names = ["Unknown"] * len(face_locations)
        
        # State machine logic
        current_time = time.time()
        if authorized_face_detected:
            if state['last_authorized'] != authorized_name:
                # New authorized user detected
                state['last_authorized'] = authorized_name
                state['access_granted_time'] = current_time
                state['showing_access_granted'] = True
                update_display(f"Access Granted", authorized=True)
                log_message(f"Access granted for {authorized_name}")
            elif state['showing_access_granted'] and (current_time - state['access_granted_time'] >= 3):
                # After 3 seconds, show Welcome <username>
                state['showing_access_granted'] = False
                update_display(f"Welcome {authorized_name}", authorized=True)
            elif not state['showing_access_granted']:
                update_display(f"Welcome {authorized_name}", authorized=True)
            RED_LED.off()
            GREEN_LED.on()
        else:
            state['last_authorized'] = None
            state['access_granted_time'] = None
            state['showing_access_granted'] = False
            RED_LED.on()
            GREEN_LED.off()
            update_display("Access Denied", authorized=False)
            if face_names:
                log_message(f"Access denied for unknown face")

        # If display is available, draw on the frame
        if has_display:
            # Draw rectangles and names on the frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled
                top *= cv_scaler
                right *= cv_scaler
                bottom *= cv_scaler
                left *= cv_scaler
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Draw a label with a name below the face
                label = name if name != "Unknown" else "Unknown"
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

        return frame
    except Exception as e:
        log_message(f"Error processing frame: {e}")
        return frame

def main():
    try:
        log_message(f"Using cv_scaler value: {cv_scaler}")
        # Initial state
        RED_LED.on()
        GREEN_LED.off()
        # Show welcome message for 3 seconds
        update_display("Welcome to Facial Recognition System", authorized=False)
        time.sleep(3)
        update_display("System Ready", authorized=False)
        log_message("Facial recognition system started")
        state = {'last_authorized': None, 'access_granted_time': None, 'showing_access_granted': False}
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while True:
            try:
                frame = picam2.capture_array()
                if frame is None:
                    raise Exception("Failed to capture frame")
                
                processed_frame = process_frame(frame, state)
                if processed_frame is None:
                    raise Exception("Failed to process frame")
                
                # If display is available, show the frame
                if has_display:
                    cv2.imshow('Video', processed_frame)
                    # Break the loop and stop the script if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Calculate and log FPS every 30 frames
                frame_count += 1
                if frame_count % 30 == 0:
                    end_time = time.time()
                    fps = 30 / (end_time - start_time)
                    log_message(f"FPS: {fps:.2f}")
                    start_time = time.time()
                
                consecutive_errors = 0  # Reset error counter on successful iteration
                time.sleep(0.1)  # Small delay to prevent overwhelming the system
                
            except Exception as e:
                consecutive_errors += 1
                log_message(f"Error in main loop iteration: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    log_message("Too many consecutive errors, restarting camera...")
                    try:
                        picam2.stop()
                        time.sleep(1)
                        picam2.start()
                        consecutive_errors = 0
                    except Exception as camera_error:
                        log_message(f"Failed to restart camera: {camera_error}")
                        raise
                time.sleep(1)  # Wait before retrying
                
    except KeyboardInterrupt:
        log_message("System stopped by user")
    except Exception as e:
        log_message(f"Fatal error in main loop: {e}")
    finally:
        # Cleanup
        try:
            RED_LED.off()
            GREEN_LED.off()
            picam2.stop()
            update_display("System Off")
            if has_display:
                cv2.destroyAllWindows()
            log_message("System shutdown complete")
        except Exception as e:
            log_message(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main() 