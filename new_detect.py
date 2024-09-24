# Import packages
import os
import cv2
import numpy as np
import importlib.util
import threading
import time

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
video = cv2.VideoCapture(1, cv2.CAP_V4L2)  # Open camera (change to video file if needed)

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

                # Draw label
                object_name = labels[int(classes[i])]  # Get object name
                label = f'{object_name}: {int(scores[i] * 100)}%'
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, label_size[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10), 
                              (xmin + label_size[0], label_ymin + base_line - 10), 
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 0, 0), 2)

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
