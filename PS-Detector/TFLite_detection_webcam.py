######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import socket
import threading

import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import ast
import queue

from pytesseract import pytesseract

# Set the path to the Tesseract executable explicitly for Windows
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(640, 480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    default='TFLite_model')
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

SERVER_IP = "127.0.0.1"
SERVER_PORT = 8900

LISTEN_IP = "127.0.0.1"
LISTEN_PORT = 8000

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if GRAPH_NAME == 'detect.tflite':
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.path.dirname(os.path.abspath(__file__))

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if 'StatefulPartitionedCall' in outname: # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2


# -----------------------------------------------------------------------------------------
# QUEUE & CACHE FOR DETECTION RESULTS
# -----------------------------------------------------------------------------------------

plate_queue = queue.Queue()
# Dictionary to store plate and the timestamp it was first seen recently
lp_history = {} 

def extract_plate_number(text):
    """Extracts exactly 7 or 8 digits from the OCR text, filtering out noise."""
    digits_only = ''.join(filter(str.isdigit, text))
    if len(digits_only) == 7 or len(digits_only) == 8:
        return digits_only
    return "Null"

def image_ch(im_np):
    # Enforce a whitelist of only numbers. Israeli plates are strictly 7-8 digits.
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    
    gray = cv2.cvtColor(im_np, cv2.COLOR_BGR2GRAY)
    
    # Resize image to 3x, making it easier for Tesseract to read from the webcam
    resized = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    # 1. First Attempt: Let Tesseract do its own automatic binarization on the resized grayscale image
    raw_text_1 = pytesseract.image_to_string(resized, config=config).strip()
    
    # 2. Second Attempt: Otsu thresholding (automatically calculates optimal threshold for lighting)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    raw_text_2 = pytesseract.image_to_string(thresh, config=config).strip()

    candidates = [raw_text_1, raw_text_2]
    best_raw = max(candidates, key=len) if any(candidates) else ""
    found_plate = "Null"
    
    for raw in candidates:
        plate = extract_plate_number(raw)
        if plate != "Null":
            found_plate = plate
            best_raw = raw
            break
            
    if plate != "Null":
        current_time = time.time()
        
        # Clean up history: remove plates seen more than 2 minutes (120 seconds) ago
        expired_plates = [p for p, t in lp_history.items() if current_time - t > 120]
        for p in expired_plates:
            del lp_history[p]
            
        # If plate is new or its 2 minute memory has expired
        if plate not in lp_history:
            lp_history[plate] = current_time
            print(f"[Detector] Found New License Plate: {plate}")
            plate_queue.put(plate)
        else:
            # We already saw this plate within the last 2 minutes, ignore it to prevent spam.
            pass

def turnOnStreamDetection():
    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    
    # Initialize video stream
    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    time.sleep(1)
    
    print("[Detector] Webcam started. Scanning for license plates...")

    while True:
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()
        if frame1 is None:
            continue

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Add 15px padding to prevent cutting off the edges!
                pad = 15
                ymin = int(max(1, (boxes[i][0] * imH) - pad))
                xmin = int(max(1, (boxes[i][1] * imW) - pad))
                ymax = int(min(imH, (boxes[i][2] * imH) + pad))
                xmax = int(min(imW, (boxes[i][3] * imW) + pad))

                cropped_lp = frame[ymin:ymax, xmin:xmax]
                
                # Check that crop size is valid
                if cropped_lp.size > 0:
                    try:
                        image_ch(cropped_lp)
                    except Exception as e:
                        print(f"[Detector] OCR error: {e}")
                else:
                    print("[Detector] Warning: Invalid crop size.")

                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        # Draw frame rate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/max(time1, 0.001)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()


# -----------------------------------------------------------------------------------------
# NETWORK PROXY
# -----------------------------------------------------------------------------------------

def query_server(msg_dict):
    """Sends a dictionary payload to the Server and returns its string response"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(5.0) # wait up to 5 seconds for the DB server to reply
    server_address = (SERVER_IP, SERVER_PORT)
    try:
        msg_str = str(msg_dict)
        sock.sendto(msg_str.encode(), server_address)
        
        server_msg, _ = sock.recvfrom(1024)
        return server_msg.decode()
    except socket.timeout:
        print("[Network] Server query timed out.")
        return "Server Timeout"
    except Exception as e:
        print(f"[Network] Server query failed: {e}")
        return "Server Error"
    finally:
        sock.close()

latest_plate_result = None
result_lock = threading.Lock()

def server_verifier_thread():
    """Runs in background: grabs plates, queries DB, and caches the string for Web Client to poll."""
    global latest_plate_result
    while True:
        try:
            detected_plate = plate_queue.get()
            print(f"[Network] Verifying new plate {detected_plate} with Server Database...")
            
            req_dict = {"Code": 200, "Data": detected_plate}
            server_response = query_server(req_dict)
            
            if "No detection in the database" in server_response:
                final_msg = f"Detection: {detected_plate} - Not Found in Database"
            else:
                final_msg = f"WARNING! Stolen License Plate detected: {detected_plate}"
                
            print(f"[Network] Verification Complete. Storing message for Web Client: '{final_msg}'")
                
            with result_lock:
                latest_plate_result = final_msg
                
        except Exception as e:
            print(f"[Network] Verifier Error: {e}")

def main_proxy():
    try:
        app_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        app_sock.bind((LISTEN_IP, LISTEN_PORT))
        print(f"[Network] Proxy listening on {LISTEN_IP}:{LISTEN_PORT} for Web Client commands...")
        app_sock.settimeout(1.0) # Enable timeout so the loop checks for CTRL+C periodically
    except Exception as e:
        print(f"[Network] Failed to bind Proxy port: {e}")
        return

    try:
        while True:
            try:
                client_msg, client_addr = app_sock.recvfrom(1024)
                msg_str = client_msg.decode()
                
                # Safe eval alternative using ast
                try:
                    app_dict = ast.literal_eval(msg_str)
                except Exception:
                    print(f"[Network] Invalid format from Web Client: {msg_str}")
                    app_sock.sendto("Error: Invalid Message Format".encode(), client_addr)
                    continue
                
                if not isinstance(app_dict, dict) or "Code" not in app_dict:
                    continue

                # ---------------------------------------------------------
                # LOGIN REQUEST (Code: 100)
                # Pass directly to Server -> Wait for Reply -> Send to Client
                # ---------------------------------------------------------
                if app_dict["Code"] == 100:
                    print(f"[Network] Web Client requesting Validation Data: Proxying Login Request...")
                    server_response = query_server(app_dict)
                    app_sock.sendto(server_response.encode(), client_addr)

                # ---------------------------------------------------------
                # DETECTION REQUEST (Code: 200)
                # Client polls for detection status. Reply instantly.
                # ---------------------------------------------------------
                elif app_dict["Code"] == 200:
                    global latest_plate_result
                    with result_lock:
                        if latest_plate_result:
                            app_sock.sendto(latest_plate_result.encode(), client_addr)
                            latest_plate_result = None # Clear after notifying client once
                        else:
                            app_sock.sendto("No detection".encode(), client_addr)
                    
            except socket.timeout:
                # 1 second passed without messages. Yield and continue.
                continue
            except Exception as e:
                print(f"[Network] Error in proxy loop: {e}")
    except KeyboardInterrupt:
        print("\n[System] CTRL+C detected. Terminating Proxy Server...")
    finally:
        app_sock.close()

def main():
    # Start Webcam processing in a background thread
    det_thread = Thread(target=turnOnStreamDetection, daemon=True)
    det_thread.start()
    
    # Start the DB Server Verifier in a background thread
    ver_thread = Thread(target=server_verifier_thread, daemon=True)
    ver_thread.start()
    
    # Run Network Proxy in main thread
    main_proxy()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[System] CTRL+C manually triggered globally. Shutting down PS-Detector...")
        sys.exit(0)