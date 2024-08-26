import io
import os
import platform
import sys
import time
import configparser
import torch
import asyncio
import cv2
import numpy as np
import importlib.util
import threading
import datetime
import customtkinter as ctk
from pathlib import Path
from PIL import Image
from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.general import cv2, non_max_suppression
from utils.torch_utils import smart_inference_mode
from aiohttp import ClientSession, FormData
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)

if platform.system() == "Windows":
    import pathlib

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"


# ----- TensorFlow Lite -----

config = configparser.ConfigParser()
config.read("config/const.ini")

# ------- CONSTANTS ---------
APP_NAME = config["DEFAULT"]["app_name"]
YOLO_WEIGHTS = config["DEFAULT"]["yolo_weights"]
MODEL_NAME = config["DEFAULT"]["model_name"]
GRAPH_NAME = config["DEFAULT"]["graph_name"]
LABELMAP_NAME = config["DEFAULT"]["labelmap_name"]
MIN_DETECT_TIME = int(config["DEFAULT"]["min_detect_time"])
CONF_THRESHOLD = float(config["DEFAULT"]["min_conf_threshold"])
SERVER_IMAGE_ENDPOINT = config["DEFAULT"]["server_image_endpoint"]
SERVER_DATA_ENDPOINT = config["DEFAULT"]["server_data_endpoint"]
ANNOTATE = config["DEFAULT"]["annotate"]
FRAME_SKIP = int(config["DEFAULT"]["frame_skip"])
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)
with open(PATH_TO_LABELS, "r") as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == "???":
    del labels[0]

pkg = importlib.util.find_spec("tflite_runtime")
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]["shape"][1]
width = input_details[0]["shape"][2]
floating_model = input_details[0]["dtype"] == np.float32
input_mean = 127.5
input_std = 127.5

# ----- Global Variables -----
global_url = None
global_device = "cpu"
global_model = None
global_show = False
global_annotation = None
global_video_label = None
global_test = True
global_log = None
global_target_detected_start_time = None
global_target = "robbery"
global_save_image = False
global_modelname = "yolo"
# --- End Global Variables ----

outname = output_details[0]["name"]

if "StatefulPartitionedCall" in outname:  # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2


def tflite_detect(frame):
    image = frame.to_ndarray(format="bgr24")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]["index"])[
        0
    ]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]["index"])[
        0
    ]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]["index"])[
        0
    ]  # Confidence of detected objects

    detections = []
    for i in range(len(scores)):
        if (scores[i] > CONF_THRESHOLD) and (scores[i] <= 1.0):
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[
                int(classes[i])
            ]  # Look up object name from "labels" array using class index
            label = "%s: %d%%" % (object_name, int(scores[i] * 100))
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )  # Get font size
            label_ymin = max(
                ymin, labelSize[1] + 10
            )  # Make sure not to draw label too close to top of window
            cv2.rectangle(
                image,
                (xmin, label_ymin - labelSize[1] - 10),
                (xmin + labelSize[0], label_ymin + baseLine - 10),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                image,
                label,
                (xmin, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
            detections.append(object_name)
    return [image, detections]


# ----- End TensorFlow Lite -----


# ----- Yolov5 -----
def yolo_detect(frame, device, model):
    image = frame.to_ndarray(format="bgr24")
    img_resized = cv2.resize(image, (320, 320))
    detections = []
    img_tensor = (
        torch.from_numpy(img_resized)
        .to(device, non_blocking=True)
        .permute(2, 0, 1)
        .unsqueeze(0)
    ).float() / 255.0
    with torch.inference_mode():
        pred = model(img_tensor)
        pred = non_max_suppression(pred)
        annotator = Annotator(image, example=str(model.names))
        label = ""
        for *xyxy, conf, cls in reversed(pred[0]):
            if conf > CONF_THRESHOLD:
                label = model.names[int(cls)]
                if ANNOTATE == "1":
                    img_label = f"{label} {conf:.2f}"
                    annotator.box_label(xyxy, img_label, color=colors(int(cls), True))
                    image = annotator.result()
                detections.append(label)
    return [image, detections]


# ----- End Yolov5 -----


async def send_data(img, label, timestamp):
    filename = f"{label}_{timestamp}.jpg"
    if SERVER_IMAGE_ENDPOINT == "":
        print("Not sending image to server")
        return

    try:
        # Convert the ndarray BGR24 image to a JPEG in memory
        _, img_encoded = cv2.imencode(".jpg", img)
        f = io.BytesIO(img_encoded.tobytes())

        async with ClientSession() as session:
            # Prepare the form data
            form = FormData()
            form.add_field(
                "image", f.getvalue(), filename=filename, content_type="image/jpeg"
            )

            # Prepare headers with additional metadata
            headers = {
                "X-Filename": filename,
                "X-Label": label,
                "X-Timestamp": timestamp,
            }

            async with session.post(
                SERVER_IMAGE_ENDPOINT, data=form, headers=headers
            ) as response:
                response.raise_for_status()
                print(await response.text())

    except Exception as e:
        print(e)



def send_data_async(img, label, timestamp):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.ensure_future(send_data(img, label, timestamp))
    else:
        loop.run_until_complete(send_data(img, label, timestamp))



@smart_inference_mode()
class VideoStreamTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        return frame


async def negotiate(pc, url):
    # Add video and audio transceivers
    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")

    # Create and set local offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # Send offer to server
    async with ClientSession() as session:
        async with session.post(
            f"{url}/offer",
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        ) as response:
            answer = await response.json()

    # Set remote description
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
    )


async def vidstream():

    pc = RTCPeerConnection()

    @pc.on("track")
    async def on_track(track):
        if track.kind == "video":
            print("Video track received")
            global global_show, global_video_label, global_test, global_annotation, global_device, global_model, global_target_detected_start_time, global_modelname, global_save_image

            video_track = VideoStreamTrack(track)
            global_log.insert("0.0", "Connected to server\n")
            print("Connected!")
            print(global_modelname)
            model = DetectMultiBackend(
                YOLO_WEIGHTS,
                global_device,
            )
            frame_count = 0
            frame_skip = FRAME_SKIP
            while True:
                try:
                    frame = await video_track.recv()
                    frame_count += 1
                    if frame_count % frame_skip == 0:
                        if global_modelname == "yolo":
                            detections = yolo_detect(frame, global_device, model)
                        else:
                            detections = tflite_detect(frame)
                        img = detections[0]
                        labels = detections[1]
                        if global_show:
                            img_pil = Image.fromarray(
                                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            )
                            img_tk = ctk.CTkImage(img_pil, size=(320, 240))
                            global_video_label.imgtk = img_tk
                            global_video_label.configure(image=img_tk)
                        else:
                            for label in labels:
                                global_log.insert(
                                    "0.0",
                                    "Detected " + label + "\n",
                                )
                        if global_target in labels:
                            if global_target_detected_start_time is None:
                                global_target_detected_start_time = time.time()
                            elif (
                                time.time() - global_target_detected_start_time
                                >= MIN_DETECT_TIME
                            ):
                                timestamp = datetime.datetime.now().strftime(
                                    "%Y-%m-%d_%H-%M-%S"
                                )
                                global_log.insert(
                                    "0.0",
                                    global_target + " detected at: " + timestamp + "\n",
                                )
                                print(
                                    global_target + " detected at: " + timestamp + "\n"
                                )
                                filename = f"{global_target}_{timestamp}.jpg"
                                send_data_async(img, global_target, timestamp)
                                if global_show == False:
                                    img_pil = Image.fromarray(
                                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    )
                                    img_tk = ctk.CTkImage(img_pil, size=(410, 310))
                                    global_video_label.imgtk = img_tk
                                    global_video_label.configure(image=img_tk)
                                if global_save_image:
                                    try:
                                        file_path = f"{detected_dir}/{filename}"
                                        cv2.imwrite(file_path, img)
                                        print("detected image saved")
                                    except Exception as e:
                                        print(f"Failed to save image: {e}")
                                global_target_detected_start_time = None
                        frame_count = 0
                    await asyncio.sleep(0.001)
                except Exception as e:
                    print("MediaStreamError occurred:", str(e))
                    break
            print("Video track ended")
            await asyncio.sleep(2)
            print("Trying to reconnect...")
            loop.call_soon_threadsafe(
                asyncio.create_task,
                vidstream(),
            )

    await negotiate(pc, global_url)
    print("Connection established!")
    while True:
        await asyncio.sleep(0.001)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(APP_NAME)
        self.geometry("725x600")

        # Frame for Video Display
        self.video_frame = ctk.CTkFrame(
            self, width=410, height=310, corner_radius=10, border_width=3
        )
        self.video_frame.place(x=19, y=20)

        # Label for Video Frame inside the frame
        self.video_label = ctk.CTkLabel(
            self.video_frame, text="", bg_color=("gray86", "gray17")
        )
        self.video_label.place(x=152, y=132)
        self.video_label.pack(fill="both", expand=True)

        # Entry for IP Address
        self.server_ip_entry = ctk.CTkEntry(self, bg_color=("gray92", "gray14"))
        self.server_ip_entry.place(x=455, y=43)

        # Label for IP Address
        self.ip_label = ctk.CTkLabel(
            self, text="IP Address:", bg_color=("gray92", "gray14")
        )
        self.ip_label.place(x=457, y=13)

        # Entry for Port
        self.port_entry = ctk.CTkEntry(self, width=99, bg_color=("gray92", "gray14"))
        self.port_entry.place(x=607, y=42)

        # Label for Port
        self.port_label = ctk.CTkLabel(
            self, text="Port:", bg_color=("gray92", "gray14")
        )
        self.port_label.place(x=610, y=14)

        # Label for Detection Target
        self.detection_label = ctk.CTkLabel(
            self, text="Detection Target:", bg_color=("gray92", "gray14")
        )
        self.detection_label.place(x=458, y=86)

        # Entry for Detection Target
        self.detection_target = ctk.CTkEntry(
            self, width=179, bg_color=("gray92", "gray14")
        )
        self.detection_target.place(x=458, y=116)

        # Label for Device
        self.device_label = ctk.CTkLabel(
            self, text="Device:", bg_color=("gray92", "gray14")
        )
        self.device_label.place(x=647, y=86)

        # Entry for Device
        self.device_entry = ctk.CTkEntry(self, width=61, bg_color=("gray92", "gray14"))
        self.device_entry.place(x=643, y=116)

        # Label for Model
        self.model_label = ctk.CTkLabel(
            self, text="Model:", bg_color=("gray92", "gray14")
        )
        self.model_label.place(x=459, y=161)

        # OptionMenu for Model Selection
        self.model_select = ctk.CTkOptionMenu(
            self,
            values=["Yolo V5", "TensorFlow Lite"],
            width=250,
            bg_color=("gray92", "gray14"),
            command=self.model_select_event,
        )
        self.model_select.place(x=455, y=192)

        # Switch for Show Video
        self.show_video_switch = ctk.CTkSwitch(
            self,
            text="Show Video",
            command=self.show_video_switch_event,
            corner_radius=99,
            border_width=0,
            bg_color=("gray92", "gray14"),
        )
        self.show_video_switch.place(x=456, y=247)

        # Switch for Save Image
        self.save_image_switch = ctk.CTkSwitch(
            self,
            text="Save Image",
            corner_radius=99,
            border_width=0,
            bg_color=("gray92", "gray14"),
            command=self.save_image_switch_event,
        )
        self.save_image_switch.place(x=588, y=246)

        # Button for Connect
        self.connect_button = ctk.CTkButton(
            self,
            text="Connect",
            command=self.connect_button_event,
            width=120,
            bg_color=("gray92", "gray14"),
        )
        self.connect_button.place(x=586, y=301)

        # Button for Disconnect
        self.disconnect_button = ctk.CTkButton(
            self,
            text="Disconnect",
            command=self.disconnect_button_event,
            width=120,
            bg_color=("gray92", "gray14"),
        )
        self.disconnect_button.place(x=453, y=302)

        # Textbox for Output Log
        self.output_log = ctk.CTkTextbox(
            self, width=689, height=206, border_width=2, bg_color=("gray92", "gray14")
        )
        self.output_log.place(x=20, y=380)

        # Label for Output Log
        self.output_label = ctk.CTkLabel(
            self, text="Output Log:", bg_color=("gray92", "gray14")
        )
        self.output_label.place(x=21, y=349)

        # Button for Clear Log
        self.clear_button = ctk.CTkButton(
            self,
            text="Clear Log",
            command=self.clear_button_event,
            width=255,
            bg_color=("gray92", "gray14"),
        )
        self.clear_button.place(x=453, y=349)

        global global_log, global_video_label
        img_pil = Image.new("RGB", (410, 310), color="black")
        img_tk = ctk.CTkImage(img_pil, size=(410, 310))
        self.video_label.imgtk = img_tk
        self.video_label.configure(image=img_tk)
        global_video_label = self.video_label
        global_log = self.output_log

        self.read_config()

    def show_video_switch_event(self):
        global global_show
        global_show = self.show_video_switch.get()
        if global_show == False:
            img_pil = Image.new("RGB", (410, 310), color="black")
            img_tk = ctk.CTkImage(img_pil, size=(410, 310))
            global_video_label.imgtk = img_tk
            global_video_label.configure(image=img_tk)

    def save_image_switch_event(self):
        global global_save_image
        global_save_image = self.save_image_switch.get()

    def model_select_event(self, selected_value):
        global global_modelname
        if selected_value == "Yolo V5":
            global_modelname = "yolo"
        else:
            global_modelname = "tflite"

    def read_config(self):
        config = configparser.ConfigParser()
        config_file = Path(__file__).parent / "config/config.ini"

        if config_file.exists():
            config.read(config_file)
            self.server_ip_entry.insert(0, config["DEFAULT"]["server_ip"])
            self.port_entry.insert(0, config["DEFAULT"]["port"])
            self.device_entry.insert(0, config["DEFAULT"]["device"])
            self.detection_target.insert(0, config["DEFAULT"]["target"])

        global global_model, global_device, global_show, global_annotation, global_test, global_video_label, global_url, global_target
        global_test = True
        global_annotation = True
        global_device = self.device_entry.get().strip()
        global_url = (
            "http://"
            + self.server_ip_entry.get().strip()
            + ":"
            + self.port_entry.get().strip()
        )
        global_show = self.show_video_switch.get()
        global_model = None
        global_target = self.detection_target.get().strip()
        return config

    def write_config(self):
        config = configparser.ConfigParser()
        config_file = Path(__file__).parent / "config/config.ini"

        if config_file.exists():
            config.read(config_file)

        config["DEFAULT"] = {
            "server_ip": self.server_ip_entry.get().strip(),
            "port": self.port_entry.get().strip(),
            "device": self.device_entry.get().strip(),
            "show_video": str(self.show_video_switch.get()),
            "target": self.detection_target.get().strip(),
        }

        with open(config_file, "w") as f:
            config.write(f)

        return config

    def connect_button_event(self):
        global global_url
        global_url = (
            "http://"
            + self.server_ip_entry.get().strip()
            + ":"
            + self.port_entry.get().strip()
        )
        self.output_log.insert(
            "end",
            f"Connecting to {global_url}\n",
        )
        self.write_config()
        self.model_select["state"] = "disabled"
        self.video_label["text"] = ""
        loop.call_soon_threadsafe(
            asyncio.create_task,
            vidstream(),
        )

    def clear_button_event(self):
        self.output_log.delete("0.0", "end")

    def disconnect_button_event(self):
        loop.create_task(self._disconnect(self))

    async def _disconnect(self, loop):
        self.output_log.insert("0.0", "Disconnecting...\n")
        try:
            async with ClientSession() as session:
                global global_url
                async with session.get(global_url + "/stop") as response:
                    response_text = await response.text()
                    self.output_log.insert("0.0", f"Server response: {response_text}\n")
                    img_pil = Image.new("RGB", (410, 310), color="black")
                    img_tk = ctk.CTkImage(img_pil, size=(410, 310))
                    global_video_label.imgtk = img_tk
                    global_video_label.configure(image=img_tk)
                    self.video_label["text"] = "Disconnected"
        except Exception as e:
            self.output_log.insert("0.0", f"Error disconnecting: {str(e)}\n")
        finally:
            # loop.call_soon_threadsafe(loop.stop)
            self.output_log.insert("0.0", "Disconnected.\n")

    def setup_buttons(self, loop):
        self.connect_button = ctk.CTkButton(
            self, text="Connect", command=self.connect_button_event
        )
        self.disconnect_button = ctk.CTkButton(
            self,
            text="Disconnect",
            command=lambda: self.disconnect_button_event(self),
        )


def start_event_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


if __name__ == "__main__":
    detected_dir = "./detected/"
    os.makedirs(detected_dir, exist_ok=True)
    app = App()
    loop = asyncio.new_event_loop()
    threading.Thread(target=start_event_loop, args=(loop,), daemon=True).start()
    app.setup_buttons(loop)
    app.mainloop()
