import cv2 
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, ClientSettings
import openpose
import argparse

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)

st.title("Object Detection")

#Select Model
st.sidebar.title("Models")
select = st.sidebar.selectbox("Select Model",['Yolov4-tiny', 'Yolov3-tiny', 'Yolov2-tiny', 'OpenPose Detection'], key='1')

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inWidth = args.width
inHeight = args.height

flag = 0
# Load Model
if select == 'Yolov4-tiny':
    net = cv2.dnn.readNet("models/yolov4-tiny.weights", "models/yolov4-tiny.cfg")
    flag = 0
elif select == 'Yolov3-tiny':
    net = cv2.dnn.readNet("models/yolov3-tiny.weights", "models/yolov3-tiny.cfg")
    flag = 0
elif select == 'Yolov2-tiny':
    net = cv2.dnn.readNet("models/yolov2-tiny.weights", "models/yolov2-tiny.cfg")
    flag = 0
elif select == 'OpenPose Detection':
    flag = 1
    
if flag == 0:
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # Loading image
            height, width, channels = img.shape
            
            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            
            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                        
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[i]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
            return img
            


    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, client_settings=WEBRTC_CLIENT_SETTINGS, async_transform=True)
elif flag == 1:
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
        
            frameHeight, frameWidth, channels = img.shape
            
            net.setInput(cv.dnn.blobFromImage(img, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
            out = net.forward()
            out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
        
            assert(len(BODY_PARTS) == out.shape[1])
        
            points = []
            for i in range(len(BODY_PARTS)):
                # Slice heatmap of corresponging body's part.
                heatMap = out[0, i, :, :]
        
                # Originally, we try to find all the local maximums. To simplify a sample
                # we just find a global one. However only a single pose at the same time
                # could be detected this way.
                _, conf, _, point = cv.minMaxLoc(heatMap)
                x = (frameWidth * point[0]) / out.shape[3]
                y = (frameHeight * point[1]) / out.shape[2]
                # Add a point if it's confidence is higher than threshold.
                points.append((int(x), int(y)) if conf > args.thr else None)
        
            for pair in POSE_PAIRS:
                partFrom = pair[0]
                partTo = pair[1]
                assert(partFrom in BODY_PARTS)
                assert(partTo in BODY_PARTS)
        
                idFrom = BODY_PARTS[partFrom]
                idTo = BODY_PARTS[partTo]
        
                if points[idFrom] and points[idTo]:
                    cv.line(img, points[idFrom], points[idTo], (0, 255, 0), 3)
                    cv.ellipse(img, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                    cv.ellipse(img, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        
            t, _ = net.getPerfProfile()
            freq = cv.getTickFrequency() / 1000
            cv.putText(img, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        
            return img

    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, client_settings=WEBRTC_CLIENT_SETTINGS, async_transform=True)
    