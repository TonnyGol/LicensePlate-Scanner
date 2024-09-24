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

from pytesseract import pytesseract


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
        Thread(target=self.update, args=()).start()
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

lp2det = []
lpCh = []

SERVER_IP = "127.0.0.1"
SERVER_PORT = 8900

LISTEN_IP = "127.0.0.1"
LISTEN_PORT = 8000

Recognition_Request = {"Code": 200, "Data": "0"}
Login_Request = {"Code": 100, "UserName": "", "PassWord": ""}

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
CWD_PATH = os.getcwd()

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

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)


def parse_output(number):
    ans = ""
    if len(number) == 7:
        return number
    elif len(number) == 8:
        return number
    elif len(number) > 8:
        for ch in number:
            if ch.isdigit():
                ans += ch
        return ans
    elif len(number) == 6:
        return "Not Rec"
    return "Not Rec"

# In[24]:
def isJustDigit(number):
    for dig in number:
        if not dig.isdigit():
            return False
    return True

def isLP(number):
    if isJustDigit(number) and (len(number) == 8 or len(number) == 7):
        return number
    return "Null"

def image_ch(im_np):
    # display(Image.fromarray(im_np))
    config = "-l eng --psm 7"
    retval, threshold = cv2.threshold(im_np, 102, 248, cv2.THRESH_BINARY)
    # display(Image.fromarray(threshold))
    gray = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)
    # display(Image.fromarray(threshold))
    blur = cv2.bilateralFilter(gray, 20, 80, 75)
    # display(Image.fromarray(blur))
    retval, threshold = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    # display(Image.fromarray(threshold))
    text = pytesseract.image_to_string(threshold, config=config)
    text = parse_output(text)
    text = isLP(text)  # here we capture the numbers in the licence plate inside text.
    # print(text)
    if not text == "Null" and text not in lp2det and text not in lpCh:
        lp2det.append(text)
        print(lp2det)
        print(lpCh)

def turnOnDetection():
    thread = Thread(target=turnOnStreamDetection, args=())
    thread.start()

def turnOnStreamDetection():
    # for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    frame_rate_calc = 0
    while True:
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

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

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))

                cropped_lp = frame[ymin:ymax, xmin:xmax]
                image_ch(cropped_lp)

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
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()


def buildRequest():
    new_msg = str(Recognition_Request)
    new_msg = eval(new_msg)
    if len(lp2det) > 0 and lp2det[0] not in lpCh:
        temp = lp2det[0]
        lpCh.append(temp)
        lp2det.remove(temp)
        new_msg["Data"] = temp
    return new_msg


def create_connect_socket():
    # ( info about the function )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return sock


def send_msg(msg, sock):
    # ( info about the function )
    msg_build = str(msg)
    server_address = (SERVER_IP, SERVER_PORT)
    sock.sendto(msg_build.encode(), server_address)


def get_msg(sock):
    # ( info about the function )
    try:
        server_msg, server_addr = sock.recvfrom(1024)
        server_msg = server_msg.decode()
        print(server_msg)
        return server_msg
    except:
        print("ERR")
        return "OK"


def close_connection(sock):
    # ( info about the function )
    sock.close()


def create_socket_for_app():
    # ( Creates a Tcp connection so clients could connect to the server )
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = (LISTEN_IP, LISTEN_PORT)
        sock.bind(server_address)
        print("Detection proxy started connection to get commands from app...")
    except:
        "An Error Happened"
        return "CONNECTION ERROR"
    else:
        return sock


def get_msg_from_app(client_soc):
    # ( server receives msg from a client and his ip address )
    try:
        client_msg, client_addr = client_soc.recvfrom(1024)
        client_msg = client_msg.decode()
        print(client_msg)
        return client_msg, client_addr
    except:
        print("ERR")
        return "OK"


def send_msg_to_app(sock, msg, client_addr):
    # ( server sends a message to the client back )
    msg_build = msg
    sock.sendto(msg_build.encode(), client_addr)


def close_socket_with_app(soc):
    # ( closes the sock so no one could connect now and the server stops )
    soc.close()


# -----------------------------------------------------------
def create_connect_socket_for_server():
    # ( info about the function )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Connected to server with Data Base")
    return sock


def build_msg_to_server(number):
    new_msg = Recognition_Request
    new_msg["Data"] = number
    return new_msg


def send_msg_to_server(msg, sock):
    # ( info about the function )
    msg_build = str(msg)
    server_address = (SERVER_IP, SERVER_PORT)
    sock.sendto(msg_build.encode(), server_address)


def get_msg_from_server(sock):
    # ( info about the function )
    try:
        server_msg, server_addr = sock.recvfrom(1024)
        server_msg = server_msg.decode()
        print(server_msg)
        return server_msg
    except:
        print("ERR")
        return "OK"


def close_connection_with_server(sock):
    # ( info about the function )
    sock.close()


def checkNumber(serverMsg):
    if serverMsg == "The number is not in the Data Base":
        return False
    return True


def try2Login(server_sock, app_sock):
    app_msg, app_addr = get_msg_from_app(app_sock)
    app_msg = eval(app_msg)
    send_msg_to_server(app_msg, server_sock)
    srvMsg = get_msg_from_server(server_sock)
    # app_msg, app_addr = get_msg_from_app(app_sock)
    # send_msg(app_msg, server_sock)
    # serverMsg = get_msg(server_sock)
    if srvMsg == "Login Bad":
        send_msg_to_app(app_sock, "Bad", app_addr)
        return True
    else:
        send_msg_to_app(app_sock, "OK", app_addr)
        return False


def doLogin(server_sock, app_sock):
    while try2Login(server_sock, app_sock):
        print("Bad login - wait for new login")


def main():
    turnOnDetection()
    app_sock = create_socket_for_app()
    server_sock = create_connect_socket_for_server()
    while True:
        msgSend = ""
        app_msg, app_addr = get_msg_from_app(app_sock)
        app_msg = eval(app_msg)
        if app_msg["Code"] == 100:
            msgSend = app_msg
        if app_msg["Code"] == 200:
            msgSend = buildRequest()

        send_msg_to_server(msgSend, server_sock)
        serverMsg = get_msg_from_server(server_sock)
        send_msg_to_app(app_sock, serverMsg, app_addr)

    # close_socket_with_app(app_sock)
    # close_connection_with_server(server_sock)


if __name__ == "__main__":
    main()