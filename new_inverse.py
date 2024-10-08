import os
import cv2
import numpy as np
import math
import threading
import time

# Specific geometry for Delta Arm:
f  =  69.0  # Base radius(mm)
rf =  88.0  # Bicep length(mm)
re = 128.0  # Forearm length(mm)
e  =  26.0  # End Effector radius(mm)

# Trigonometric constants
sqrt3  = math.sqrt(3.0)
pi     = 3.141592653
sin120 = sqrt3 / 2.0
cos120 = -0.5
tan60  = sqrt3
sin30  = 0.5
tan30  = 1.0 / sqrt3

# Delta arm inverse kinematics functions
def forward(theta1, theta2, theta3):
    x0 = 0.0
    y0 = 0.0
    z0 = 0.0
    
    t = (f-e) * tan30 / 2.0
    dtr = pi / 180.0
    
    theta1 *= dtr
    theta2 *= dtr
    theta3 *= dtr
    
    y1 = -(t + rf * math.cos(theta1))
    z1 = -rf * math.sin(theta1)
    
    y2 = (t + rf * math.cos(theta2)) * sin30
    x2 = y2 * tan60
    z2 = -rf * math.sin(theta2)
    
    y3 = (t + rf * math.cos(theta3)) * sin30
    x3 = -y3 * tan60
    z3 = -rf * math.sin(theta3)
    
    dnm = (y2 - y1) * x3 - (y3 - y1) * x2
    
    w1 = y1 * y1 + z1 * z1
    w2 = x2 * x2 + y2 * y2 + z2 * z2
    w3 = x3 * x3 + y3 * y3 + z3 * z3
    
    a1 = (z2 - z1) * (y3 - y1) - (z3 - z1) * (y2 - y1)
    b1 = -((w2 - w1) * (y3 - y1) - (w3 - w1) * (y2 - y1)) / 2.0
    
    a2 = -(z2 - z1) * x3 + (z3 - z1) * x2
    b2 = ((w2 - w1) * x3 - (w3 - w1) * x2) / 2.0
    
    a = a1 * a1 + a2 * a2 + dnm * dnm
    b = 2.0 * (a1 * b1 + a2 * (b2 - y1 * dnm) - z1 * dnm * dnm)
    c = (b2 - y1 * dnm) * (b2 - y1 * dnm) + b1 * b1 + dnm * dnm * (z1 * z1 - re * re)
    
    d = b * b - 4.0 * a * c
    if d < 0.0:
        return [1, 0, 0, 0]  # non-existing point
    
    z0 = -0.5 * (b + math.sqrt(d)) / a
    x0 = (a1 * z0 + b1) / dnm
    y0 = (a2 * z0 + b2) / dnm

    return [0, x0, y0, z0]

def angle_yz(x0, y0, z0, theta=None):
    y1 = -0.5 * 0.57735 * f
    y0 -= 0.5 * 0.57735 * e
    a = (x0 * x0 + y0 * y0 + z0 * z0 + rf * rf - re * re - y1 * y1) / (2.0 * z0)
    b = (y1 - y0) / z0

    d = -(a + b * y1) * (a + b * y1) + rf * (b * b * rf + rf)
    if d < 0:
        return [1, 0]

    yj = (y1 - a * b - math.sqrt(d)) / (b * b + 1)
    zj = a + b * yj
    theta = math.atan(-zj / (y1 - yj)) * 180.0 / pi + (180.0 if yj > y1 else 0.0)
    
    return [0, theta]

def inverse(x0, y0, z0):
    theta1 = 0
    theta2 = 0
    theta3 = 0
    status = angle_yz(x0, y0, z0)

    if status[0] == 0:
        theta1 = status[1]
        status = angle_yz(x0 * cos120 + y0 * sin120, y0 * cos120 - x0 * sin120, z0, theta2)
    if status[0] == 0:
        theta2 = status[1]
        status = angle_yz(x0 * cos120 - y0 * sin120, y0 * cos120 + x0 * sin120, z0, theta3)
    theta3 = status[1]

    return [status[0], theta1, theta2, theta3]

# Set fixed model parameters
MODEL_NAME = 'custom_model_lite'
GRAPH_NAME = 'edgetpu.tflite'
LABELMAP_NAME = 'labelmap.txt'
min_conf_threshold = 0.5 

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file (model)
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Remove the first label if it is '???' (COCO model specific)
if labels[0] == '???':
    del(labels[0])

# Load the TensorFlow Lite model with TPU delegate
from tflite_runtime.interpreter import Interpreter, load_delegate
interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# Model output indexing for TensorFlow version differences
outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Open video capture stream from camera
video = cv2.VideoCapture(1, cv2.CAP_V4L2)

# Set camera resolution to 640x480 and 30fps
video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video.set(cv2.CAP_PROP_FPS, 30)

imW = 640
imH = 480

# Use threading to separate capture from detection
frame_ready = False
frame = None
detection_result = None

def process_detection():
    global detection_result, frame_ready, frame

    while True:
        if frame_ready:
            # Prepare frame for detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if needed
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
            classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
            scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

            detection_result = (boxes, classes, scores)
            frame_ready = False

# Start the detection thread
thread = threading.Thread(target=process_detection)
thread.start()

while video.isOpened():
    start_time = time.time()

    ret, frame = video.read()
    if not ret:
        print('Reached the end of the video!')
        break

    frame_ready = True

    if detection_result:
        boxes, classes, scores = detection_result

        # Loop over detection results and draw boxes/labels
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)

                # Compute the center of the bounding box
                x_center = int((xmin + xmax) / 2)
                y_center = int((ymin + ymax) / 2)
                z_center = 100  # Set z to a fixed value

                # Draw center point and lines
                cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)

                # Call inverse kinematics function
                status, theta1, theta2, theta3 = inverse(x_center, y_center, z_center)

                # Prepare text for center coordinates and angles
                coord_text = f'x(center): {x_center}\ny(center): {y_center}\nz(center): {z_center}'
                angles_text = f'theta1: {theta1:.2f}\ntheta2: {theta2:.2f}\ntheta3: {theta3:.2f}'

                # Display the text in the top-left corner of the frame
                complete_text = f'{coord_text}\n{angles_text}'
                y_offset = 30  # Initial vertical offset for the text
                for line in complete_text.split('\n'):
                    cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 30  # Adjust offset for the next line

    # Display frame
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

    # FPS calculation
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    print(f"FPS: {fps:.2f}")

# Clean up
video.release()
cv2.destroyAllWindows()
