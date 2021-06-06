import cv2 
import numpy as np
import streamlit as st
from typing import List
from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)
import argparse
import av
import pydub
import asyncio
from pathlib import Path
import urllib.request
import time

HERE = Path(__file__).parent

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

# https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3
MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm"  # noqa
LANG_MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer"  # noqa
MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.pbmm"
LANG_MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.scorer"

download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=188915987)
download_file(LANG_MODEL_URL, LANG_MODEL_LOCAL_PATH, expected_size=953363776)

st.sidebar.title("Programs")
select = st.sidebar.selectbox("Select Program",['Object Detection', 'OpenPose Detection', 'Speech to Text'], key='1')

flag = 0

if select == 'Object Detection':
    flag = 0
elif select == 'OpenPose Detection':
    flag = 1
elif select == 'Speech to Text':
    flag = 2
    
if flag == 0:
    st.title("Object Detection")
    net = cv2.dnn.readNet("models/yolov4-tiny.weights", "models/yolov4-tiny.cfg")
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
            


    webrtc_ctx = webrtc_streamer(key="object filter", video_transformer_factory=VideoTransformer, client_settings=WEBRTC_CLIENT_SETTINGS, async_transform=True)
elif flag == 1:

    st.title("Pose Detection")
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
    
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

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
        
            frameHeight, frameWidth, channels = img.shape
            
            net.setInput(cv2.dnn.blobFromImage(img, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
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
                _, conf, _, point = cv2.minMaxLoc(heatMap)
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
                    cv2.line(img, points[idFrom], points[idTo], (0, 255, 0), 3)
                    cv2.ellipse(img, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                    cv2.ellipse(img, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
        
            t, _ = net.getPerfProfile()
            freq = cv2.getTickFrequency() / 1000
            cv2.putText(img, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        
            return img

    webrtc_ctx = webrtc_streamer(key="openpose filter", video_transformer_factory=VideoTransformer, client_settings=WEBRTC_CLIENT_SETTINGS, async_transform=True)

elif 2:
    st.title("Speech to Text")

    class AudioProcessor(AudioProcessorBase):

        async def recv_queued(self, frames: List[av.AudioFrame]) -> av.AudioFrame:
            with self.frames_lock:
                self.frames.extend(frames)

            # Return empty frames to be silent.
            new_frames = []
            for frame in frames:
                input_array = frame.to_ndarray()
                new_frame = av.AudioFrame.from_ndarray(
                    np.zeros(input_array.shape, dtype=input_array.dtype),
                    layout=frame.layout.name,
                )
                new_frame.sample_rate = frame.sample_rate
                new_frames.append(new_frame)

            return new_frames
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text-w-video",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        client_settings=ClientSettings(
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": True},
        ),
    )
    while True:
        lm_alpha = 0.931289039105002
        lm_beta = 1.1834137581510284
        beam = 100
        status_indicator = st.empty()
        text_output = st.empty()

        if webrtc_ctx.audio_processor:
            if stream is None:
                from deepspeech import Model

                model = Model(MODEL_LOCAL_PATH)
                model.enableExternalScorer(LANG_MODEL_LOCAL_PATH)
                model.setScorerAlphaBeta(lm_alpha, lm_beta)
                model.setBeamWidth(beam)

                stream = model.createStream()

                status_indicator.write("Model loaded.")

            sound_chunk = pydub.AudioSegment.empty()

            audio_frames = []
            with webrtc_ctx.audio_processor.frames_lock:
                while len(webrtc_ctx.audio_processor.frames) > 0:
                    frame = webrtc_ctx.audio_processor.frames.popleft()
                    audio_frames.append(frame)

            if len(audio_frames) == 0:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                    model.sampleRate()
                )
                buffer = np.array(sound_chunk.get_array_of_samples())
                stream.feedAudioContent(buffer)
                text = stream.intermediateDecode()
                text_output.markdown(f"**Text:** {text}")
        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break