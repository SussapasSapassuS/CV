# Import packages
import os
import cv2
import numpy as np
import importlib.util
import threading
import time
import math

# Set fixed model parameters
MODEL_NAME = 'custom_model_lite'
GRAPH_NAME = 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
LABELMAP_NAME = 'coco_labels.txt'
min_conf_threshold = 0.4

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file (model)
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as read_label:
    labels = [line.strip() for line in read_label.readlines()]

labels_mapping = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    12: "stop sign",
    13: "parking meter",
    14: "bench",
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    19: "sheep",
    20: "cow",
    21: "elephant",
    22: "bear",
    23: "zebra",
    24: "giraffe",
    26: "backpack",
    27: "umbrella",
    30: "handbag",
    31: "tie",
    32: "suitcase",
    33: "frisbee",
    34: "skis",
    35: "snowboard",
    36: "sports ball",
    37: "kite",
    38: "baseball bat",
    39: "baseball glove",
    40: "skateboard",
    41: "surfboard",
    42: "tennis racket",
    43: "bottle",
    45: "wine glass",
    46: "cup",
    47: "fork",
    48: "knife",
    49: "spoon",
    50: "bowl",
    51: "banana",
    52: "apple",
    53: "sandwich",
    54: "orange",
    55: "broccoli",
    56: "carrot",
    57: "hot dog",
    58: "pizza",
    59: "donut",
    60: "cake",
    61: "chair",
    62: "couch",
    63: "potted plant",
    64: "bed",
    66: "dining table",
    69: "toilet",
    71: "tv",
    72: "laptop",
    73: "mouse",
    74: "remote",
    75: "keyboard",
    76: "cell phone",
    77: "microwave",
    78: "oven",
    79: "toaster",
    80: "sink",
    81: "refrigerator",
    83: "book",
    84: "clock",
    85: "vase",
    86: "scissors",
    87: "teddy bear",
    88: "hair drier",
    89: "toothbrush"
}

# Remove the first label if it is '???' (COCO model specific)
if labels[0] == '???':
    del (labels[0])

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

# Specific geometry for Delta Arm:
f = 69.0  # Base radius(mm)
rf = 88.0  # Bicep length(mm)
re = 128.0  # Forearm length(mm)
e = 26.0  # End Effector radius(mm)

# Trigonometric constants
sqrt3 = math.sqrt(3.0)
pi = 3.141592653
sin120 = sqrt3 / 2.0
cos120 = -0.5
tan60 = sqrt3
sin30 = 0.5
tan30 = 1.0 / sqrt3


# Function: Forward Kinematics
def forward(theta1, theta2, theta3):
    t = (f - e) * tan30 / 2.0
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


# Function: Inverse Kinematics
def angle_yz(x0, y0, z0, theta=None):
    y1 = -0.5 * 0.57735 * f
    y0 -= 0.5 * 0.57735 * e
    a = (x0 * x0 + y0 * y0 + z0 * z0 + rf * rf - re * re - y1 * y1) / (2.0 * z0)
    b = (y1 - y0) / z0

    d = -(a + b * y1) * (a + b * y1) + rf * (b * b * rf + rf)
    if d < 0:
        return [1, 0]  # non-existing

    yj = (y1 - a * b - math.sqrt(d)) / (b * b + 1)
    zj = a + b * yj
    theta = math.atan(-zj / (y1 - yj)) * 180.0 / pi + (180.0 if yj > y1 else 0.0)
    return [0, theta]


def inverse(x0, y0, z0):
    status, theta1 = angle_yz(x0, y0, z0)
    if status == 0:
        status, theta2 = angle_yz(x0 * cos120 + y0 * sin120, y0 * cos120 - x0 * sin120, z0)
    if status == 0:
        status, theta3 = angle_yz(x0 * cos120 - y0 * sin120, y0 * cos120 + x0 * sin120, z0)
    return [status, theta1, theta2, theta3]


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
        print('Cannot Open Camera')
        break

    frame_ready = True

    if detection_result:
        boxes, classes, scores = detection_result

        # Loop over detection results and draw boxes/labels
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                if int(classes[i]) == 76:
                    # Get bounding box coordinates
                    ymin = int(max(1, (boxes[i][0] * imH)))
                    xmin = int(max(1, (boxes[i][1] * imW)))
                    ymax = int(min(imH, (boxes[i][2] * imH)))
                    xmax = int(min(imW, (boxes[i][3] * imW)))

                    # Draw bounding box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)

                    # Calculate center of the bounding box
                    x_center = (xmin + xmax) // 2
                    y_center = (ymin + ymax) // 2

                    # Draw center point
                    cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)  # Red circle at the center
                    
                    # Draw lines from center point to edges of the bounding box
                    cv2.line(frame, (x_center, y_center), (x_center, ymin), (255, 0, 0), 2)  # Vertical line to top
                    cv2.line(frame, (x_center, y_center), (xmin, y_center), (0, 255, 0), 2)  # Horizontal line to left

                    # Draw detection range (a rectangle representing the bounding box)
                    detection_range_color = (255, 0, 255)  # Purple color for detection range
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), detection_range_color, 2)
                    
                    # Display label with percentage confidence
                    object_name = labels_mapping.get(76, "Unknown") 
                    label = '%s: %.2f%%' % (object_name, scores[i] * 100)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), 
                                (xmin + labelSize[0], label_ymin + baseLine - 10), 
                                (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin - 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    # Calculate center coordinates for IK
                    x_center = (xmin + xmax) / 2.0
                    y_center = (ymin + ymax) / 2.0
                    x_center = x_center - imW / 2.0
                    y_center = imH / 2.0 - y_center
                    z_center = 100.0  # Example z value for the inverse kinematics

                    # Calculate inverse kinematics for the center point
                    status, theta1, theta2, theta3 = inverse(x_center, y_center, z_center)
                    if status == 0:
                        # Define some colors and fonts for better styling
                        text_color = (255, 255, 255)  # White color for text
                        bg_color = (0, 0, 0)  # Black color for background rectangle
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2

                        # Stylish text for input coordinates and z value
                        cv2.rectangle(frame, (10, 20), (250, 200), bg_color, cv2.FILLED)
                        cv2.putText(frame, f'Input x: {x_center:.2f}', (20, 40), font, font_scale, text_color, thickness, cv2.LINE_AA)
                        cv2.putText(frame, f'Input y: {y_center:.2f}', (20, 70), font, font_scale, text_color, thickness, cv2.LINE_AA)
                        cv2.putText(frame, f'Input z: {z_center:.2f}', (20, 100), font, font_scale, text_color, thickness, cv2.LINE_AA)

                        # Stylish text for theta angles
                        cv2.rectangle(frame, (10, 110), (250, 200), bg_color, cv2.FILLED)
                        cv2.putText(frame, f'Theta1: {theta1:.2f} deg', (20, 130), font, font_scale, text_color, thickness, cv2.LINE_AA)
                        cv2.putText(frame, f'Theta2: {theta2:.2f} deg', (20, 160), font, font_scale, text_color, thickness, cv2.LINE_AA)
                        cv2.putText(frame, f'Theta3: {theta3:.2f} deg', (20, 190), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Display FPS on the frame
    fps = 1.0 / (time.time() - start_time)

    # Get the text size to position it correctly
    fps_text = "FPS: %.2f" % fps
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    # Calculate the size of the text
    (text_width, text_height), baseline = cv2.getTextSize(fps_text, font, font_scale, font_thickness)

    # Calculate the position for bottom right corner
    x = frame.shape[1] - text_width - 30  # 30 pixels from the right
    y = frame.shape[0] - text_height - 10  # 10 pixels from the bottom

    # Display FPS on the frame at the bottom right
    cv2.putText(frame, fps_text, (x, y), font, font_scale, (255, 255, 0), font_thickness, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
