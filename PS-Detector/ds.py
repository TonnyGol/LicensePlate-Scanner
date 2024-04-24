import os
import pathlib

os.system('pip install --user -U --pre tensorflow=="2.*"')
os.system('pip install --user pycocotools')
os.system('pip install --user pytesseract')
os.system('pip install --user tesseract')
#------------------------------------------



if "models" in pathlib.Path.cwd().parts:
    while "models" in pathlib.Path.cwd().parts:
        os.chdir('..')
elif not pathlib.Path('models').exists():
    os.system('git clone --depth 1 https://github.com/tensorflow/models')
os.system( 'cd models/research/\n/home/magshimim/Desktop/anpr/models/research/bin/protoc object_detection/protos/*.proto --python_out=.')
os.system('cd models/research\npip install --user .')


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import pytesseract
import socket
import sqlite3
import threading

from sqlite3 import Error
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from threading import Thread, Lock



# Define
cap = cv2.VideoCapture(0)

# Import the object detection module.

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Patches:

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

#Varablie

lp2det = []
lpCh =[]
SERVER_IP = "127.0.0.1"
SERVER_PORT = 8900
LISTEN_IP = "127.0.0.1"
LISTEN_PORT = 8000
Recognition_Request = {"Code":200, "Data":"0"}
Login_Request = {"Code":100, "UserName":"", "PassWord":""}
PATH_TO_CKPT = '/home/magshimim/Desktop/anpr/models/research/object_detection/anpr_graph1/frozen_inference_graph.pb'
PATH_TO_LABELS = 'models/research/object_detection/training/object-detection.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
model_name = 'anpr_graph1'
IMAGE_SIZE = (12, 8)
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
          

def load_model():
    model_dir = '/home/magshimim/Desktop/anpr/models/research/object_detection/anpr_graph1/saved_model'
    model = tf.saved_model.load(str(model_dir),None)
    model = model.signatures['serving_default']
    return model


def make_model():
    detection_model = load_model()



def praseStr(number):
    numbers = []
    ans = ""
    for ch in number:
        if ch.isdigit():
            ans += ch
        elif ch.isalpha():
            for digit in numbers_like_ABC.keys():
                if ch in  numbers_like_ABC[digit]:
                    ans += ch
    return ans




def isReaded(number_list):
    numbers = [] 
    for number in number_list:
        if len(number) == 8 or len(number) == 7:
            numbers.append(number)
        elif len(number) == 10:
            numbers.append(number[1:-1])
        elif len(number) == 9:
            if number[0].isupper():
                numbers.append(number[1:])
            elif number[-1].isupper():
                numbers.append(number[:-1])
            else:
                numbers.append(number[:-1])
                numbers.append(number[1:])
    return numbers



def prase_output(number):
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
    #display(Image.fromarray(im_np)) 
    config = ("-l eng --psm 7")
    retval, threshold = cv2.threshold(im_np, 102, 248, cv2.THRESH_BINARY)
    #display(Image.fromarray(threshold)) 
    gray = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)
    #display(Image.fromarray(threshold)) 
    blur = cv2.bilateralFilter(gray, 20, 80, 75)
    #display(Image.fromarray(blur))
    retval, threshold = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    #display(Image.fromarray(threshold)) 
    text = pytesseract.image_to_string(threshold, config=config)
    text = prase_output(text)
    text = isLP(text)
    if not text == "Null" and text not in lp2det and text not in lpCh:
        print (text)
        lp2det.append(text)


def cropLP(image_np, boxes):
    (frame_height, frame_width) = image_np.shape[:2]
    box = (np.squeeze(boxes)[0])
    ymin = int((box[0]*frame_height))
    xmin = int((box[1]*frame_width)) + 5
    ymax = int((box[2]*frame_height))
    xmax = int((box[3]*frame_width))
    cropped_img = image_np[ymin:ymax,xmin:xmax]
    return cropped_img

def turnOnStreamDetection():
    #dg()O
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            while True:
                ret, image_np = cap.read()
                #Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
                if ((np.squeeze(scores)[0]) > 0.82):
                    lp = cropLP(image_np, boxes)
                    image_ch(lp)
                cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

def turnOnDetection():
    p = threading.Thread(target = turnOnStreamDetection, args = ())
    p.start()
    
def buildRequest():
    new_msg = str(Recognition_Request)
    new_msg = eval(new_msg)
    if len(lp2det)>0 and lp2det[0] not in lpCh:
        temp = lp2det[0]
        print (temp)
        lpCh.append(temp)
        lp2det.remove(temp)
        new_msg["Data"] = temp
    return new_msg

def create_connect_socket():
#( info about the function )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return sock


    
def send_msg(msg, sock):
#( info about the function )
    msg_build = str(msg)
    server_address = (SERVER_IP, SERVER_PORT)
    sock.sendto(msg_build.encode(), server_address)
    
def get_msg(sock):
#( info about the function )
    try:
        server_msg, server_addr = sock.recvfrom(1024)
        server_msg = server_msg.decode()
        print(server_msg)
        return server_msg
    except:
        print("ERR")
        return "OK"

def close_connection(sock):
#( info about the function )
    sock.close()




        
def create_socket_for_app():
#( Creates a Tcp connection so clients could connect to the server )
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = (LISTEN_IP, LISTEN_PORT)
        sock.bind(server_address)
    except:
        "An Error Happened"
        return "CONNECTION ERROR"
    else:
        return sock

def get_msg_from_app(client_soc):
#( server receives msg from a client and his ip address )
    try:
        client_msg, client_addr = client_soc.recvfrom(1024)
        client_msg = client_msg.decode()
        print(client_msg)
        return client_msg, client_addr
    except:
        print("ERR")
        return "OK"

def send_msg_to_app(sock, msg, client_addr):
#( server sends an message to the client back )
    msg_build = msg
    sock.sendto(msg_build.encode(), client_addr)
       
def close_socket_with_app(soc):
#( closes the sock so no one could connect now and the server stops )
    soc.close()
#-----------------------------------------------------------
def create_connect_socket_for_server():
#( info about the function )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return sock

def build_msg_to_server(number):

    new_msg = Recognition_Request
    new_msg["Data"] = number
    return new_msg

def send_msg_to_server(msg, sock):
#( info about the function )
    msg_build = str(msg)
    server_address = (SERVER_IP, SERVER_PORT)
    sock.sendto(msg_build.encode(), server_address)

def get_msg_from_server(sock):
#( info about the function )
    try:
        server_msg, server_addr = sock.recvfrom(1024)
        server_msg = server_msg.decode()
        print(server_msg)
        return server_msg
    except:
        print("ERR")
        return "OK"

def close_connection_with_server(sock):
#( info about the function )
    sock.close()


def checkNumber(serverMsg):
    if  serverMsg == "The number is not in the Data Base":
        return False
    return True

def try2Login(server_sock, app_sock):
    app_msg, app_addr = get_msg_from_app(app_sock)
    app_msg = eval(app_msg)
    send_msg_to_server(app_msg, server_sock)
    srvMsg = get_msg_from_server(server_sock)
    #app_msg, app_addr = get_msg_from_app(app_sock)
    #send_msg(app_msg, server_sock)
    #serverMsg = get_msg(server_sock)
    if srvMsg == "Login Bad":
        send_msg_to_app(app_sock, "Bad", app_addr)
        return True
    else:
        send_msg_to_app(app_sock, "OK", app_addr)
        return False
    
def doLogin(server_sock, app_sock):
    while try2Login(server_sock, app_sock):
        print ("Bad login - wait for new login")
        
 
def main():
    turnOnDetection()
    app_sock = create_socket_for_app()
    server_sock = create_connect_socket_for_server()
    while True:
        msgSend = ""
        app_msg , app_addr = get_msg_from_app(app_sock)
        app_msg = eval(app_msg)
        if app_msg["Code"] == 100:
            msgSend = app_msg
        if app_msg["Code"] == 200:
            msgSend = buildRequest()
            
        send_msg_to_server(msgSend, server_sock)
        serverMsg = get_msg_from_server(server_sock)
        send_msg_to_app(app_sock, serverMsg, app_addr)

    close_socket_with_app(app_sock)
    close_connection_with_server(server_sock)
    
if __name__ == "__main__": 
    main()

