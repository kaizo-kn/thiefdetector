import argparse
import os
import platform
import sys
from pathlib import Path
import time
import torch
import asyncio
import cv2
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiohttp import ClientSession
import customtkinter
import threading
import datetime
from PIL import Image, ImageTk

if platform.system() == "Windows":
    import pathlib

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.general import (
    cv2,
    non_max_suppression,
    scale_boxes,
)
from utils.torch_utils import select_device, smart_inference_mode

customtkinter.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"


global_url = None
global_device = None
global_model = None
global_show = None
global_annotation = None
global_video_label = None
global_test = None
global_log = None
global_person_detected_start_time = None


@smart_inference_mode()
class VideoStreamTrack(VideoStreamTrack):
    def __init__(self, track, model, device, annotation, test):
        super().__init__()
        self.track = track
        self.model = model
        self.device = device
        self.annotation = annotation
        self.test = test

    async def recv(self):
        frame = await self.track.recv()
        if self.test == True:
            return frame
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (320, 320))
        img_tensor = (
            torch.from_numpy(img_resized)
            .to(self.device, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
        ).float() / 255.0  # Normalize to [0, 1]

        with torch.inference_mode():
            pred = self.model(img_tensor)
            pred = non_max_suppression(pred)

        person_detected = False
        if self.annotation == True:
            annotator = Annotator(img, example=str(self.model.names))
            for *xyxy, conf, cls in reversed(pred[0]):
                label = f"{self.model.names[int(cls)]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(int(cls), True))
                if self.model.names[int(cls)] == "person":
                    person_detected = True
                    global_log.insert(
                        "0.0", f"Model detected person with confidence {conf:.2f}\n"
                    )
            img = annotator.im
        else:
            for det in pred:
                for *_, conf, cls in reversed(det):
                    print(
                        f"Model detected {self.model.names[int(cls)]} with confidence {conf:.2f}"
                    )
                    if self.model.names[int(cls)] == "person":
                        person_detected = True
        if person_detected:
            global global_person_detected_start_time, global_video_label, global_show
            if global_person_detected_start_time is None:
                global_person_detected_start_time = time.time()
            elif time.time() - global_person_detected_start_time >= 3:
                global_log.insert(
                    "0.0", "Person detected at: " + str(datetime.datetime.now()) + "\n"
                )
                if global_show == False:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    img_tk = ImageTk.PhotoImage(img_pil)
                    global_video_label.imgtk = img_tk
                    global_video_label.configure(image=img_tk)
                    global_person_detected_start_time = None
                    await asyncio.sleep(0.02)
                cv2.imwrite(
                    f"./detected/person_{str(datetime.datetime.now()).replace(' ', '_')}.jpg", img
                )
        else:
            global_person_detected_start_time = None
        return img


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
    global global_url, global_model, global_show, global_annotation, global_video_label, global_test

    pc = RTCPeerConnection()

    @pc.on("track")
    async def on_track(track):
        if track.kind == "video":
            print("Video track received")
            global global_show, global_video_label, global_test, global_annotation, global_device, global_model
            video_track = VideoStreamTrack(
                track, global_model, global_device, global_annotation, global_test
            )
            print("Connected!")
            while True:
                try:
                    img = await video_track.recv()
                    if global_show:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_rgb)
                        img_tk = ImageTk.PhotoImage(img_pil)
                        global_video_label.imgtk = img_tk
                        global_video_label.configure(image=img_tk)
                        await asyncio.sleep(0.02)
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
        await asyncio.sleep(0.3)


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("SABER PKM")
        self.geometry(f"{800}x400")
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure((0, 1), weight=1)
        self.grid_rowconfigure(2, weight=0)

        self.video_frame = customtkinter.CTkFrame(self, corner_radius=10)
        self.video_frame.grid(
            row=0, column=0, rowspan=2, columnspan=2, padx=20, pady=20, sticky="nsew"
        )
        self.video_label = customtkinter.CTkLabel(
            self.video_frame, text="", font=customtkinter.CTkFont(size=20)
        )
        self.video_label.pack(expand=True)

        self.server_ip_entry = customtkinter.CTkEntry(
            self, placeholder_text="Server IP"
        )
        self.server_ip_entry.grid(row=0, column=2, padx=20, pady=(20, 10), sticky="ew")

        self.port_entry = customtkinter.CTkEntry(self, placeholder_text="Port")
        self.port_entry.grid(row=1, column=2, padx=20, pady=(10, 10), sticky="ew")

        self.args_entry = customtkinter.CTkTextbox(self, height=100)
        self.args_entry.grid(row=2, column=2, padx=20, pady=(10, 20), sticky="ew")

        # Disconnect button on the left
        self.disconnect_button = customtkinter.CTkButton(
            self, text="Disconnect", command=self.disconnect_button_event
        )
        self.disconnect_button.grid(
            row=3, column=1, padx=10, pady=(10, 20), sticky="ew"
        )

        # Connect button on the right
        self.connect_button = customtkinter.CTkButton(
            self, text="Connect", command=self.connect_button_event
        )
        self.connect_button.grid(row=3, column=2, padx=10, pady=(10, 20), sticky="ew")

        self.output_log = customtkinter.CTkTextbox(self, height=100)
        self.output_log.grid(
            row=2, column=0, columnspan=1, padx=20, pady=(10, 20), sticky="nsew"
        )
        self.server_ip_entry.insert(0, "127.0.0.1")
        self.port_entry.insert(0, "3123")
        global global_log
        global_log = self.output_log

    def connect_button_event(self):
        server_ip = self.server_ip_entry.get()
        port = self.port_entry.get()
        args = self.args_entry.get("0.0", "end").strip()
        self.url = f"http://{server_ip}:{port}"
        self.output_log.insert("end", f"Connecting to {self.url} with args: {args}\n")
        # Parse the args from the text entry
        parser = argparse.ArgumentParser(
            description="YOLOv5 Real-time Object Detection"
        )
        parser.add_argument(
            "--device", type=str, default="cpu", help="Device to run inference on"
        )
        parser.add_argument(
            "--model", type=str, default="yolov5s.pt", help="Model path or name"
        )
        parser.add_argument(
            "--show", action="store_true", default="false", help="Show inference output"
        )
        parser.add_argument(
            "--annotation",
            type=str,
            default="false",
            help="Show annotation on inference output",
        )

        global global_model, global_device, global_show, global_annotation, global_test, global_video_label, global_url
        # Update the arguments based on user input
        args_list = args.split()
        known_args, unknown_args = parser.parse_known_args(args_list)
        global_test = False
        global_annotation = True
        global_device = known_args.device
        global_url = self.url
        global_video_label = self.video_label
        global_show = False
        global_model = DetectMultiBackend(
            known_args.model,
            known_args.device,
        )

        loop.call_soon_threadsafe(
            asyncio.create_task,
            vidstream(),
        )

    def disconnect_button_event(self, loop):
        loop.create_task(self._disconnect(loop))

    async def _disconnect(self, loop):
        self.output_log.insert("end", "Disconnecting...\n")
        try:
            async with ClientSession() as session:
                async with session.get(self.url + "/stop") as response:
                    response_text = await response.text()
                    self.output_log.insert("end", f"Server response: {response_text}\n")
        except Exception as e:
            self.output_log.insert("end", f"Error disconnecting: {str(e)}\n")
        finally:
            loop.call_soon_threadsafe(loop.stop)
            self.output_log.insert("end", "Disconnected.\n")

    def setup_buttons(self, loop):
        self.connect_button = customtkinter.CTkButton(
            self, text="Connect", command=self.connect_button_event
        )
        self.connect_button.grid(row=3, column=2, padx=20, pady=(10, 20), sticky="ew")

        self.disconnect_button = customtkinter.CTkButton(
            self, text="Disconnect", command=lambda: self.disconnect_button_event(loop)
        )
        self.disconnect_button.grid(
            row=3, column=1, padx=20, pady=(10, 20), sticky="ew"
        )


def start_event_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


if __name__ == "__main__":
    app = App()
    loop = asyncio.new_event_loop()
    threading.Thread(target=start_event_loop, args=(loop,), daemon=True).start()
    app.setup_buttons(loop)
    app.mainloop()
