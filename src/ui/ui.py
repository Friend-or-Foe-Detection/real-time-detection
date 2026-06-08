import customtkinter as ctk
from customtkinter import CTkImage
import tkinter as tk
from tkinter import ttk
from tkintermapview import TkinterMapView
import socket
import threading
import time
from PIL import Image, ImageDraw, ImageTk
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class MilitaryGPSVisualizer(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ---- Window Configuration ----
        ctk.set_appearance_mode("dark")
        self.title("MILITARY-GRADE GPS & YOLO Tracker")
        self.geometry("1440x900")
        self.configure(fg_color="black")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ---- Paned Window for dynamic resizing ----
        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.grid(row=0, column=0, sticky="nsew")

        # Left pane for video
        self.left_frame = tk.Frame(paned, bg="black")
        paned.add(self.left_frame, weight=1)
        # Right pane for map
        self.right_frame = tk.Frame(paned, bg="#151515")
        paned.add(self.right_frame, weight=1)

        # ---- Video Display (Left) with fixed size ----
        self.video_label = ctk.CTkLabel(self.left_frame, text="", width=640, height=480)
        self.video_label.pack(anchor="center", padx=5, pady=5)
        self.cam_gps_label = ctk.CTkLabel(self.left_frame, text="Camera GPS: N/A", font=("Consolas", 14), text_color="#FFFFFF")
        self.cam_gps_label.pack(anchor="w", padx=5, pady=2)

        # ---- Map Widget (Right) ----
        self.map_widget = TkinterMapView(self.right_frame)
        self.map_widget.pack(expand=True, fill="both", padx=5, pady=5)
        self.map_widget.set_zoom(15)
        self.map_widget.set_tile_server(
            "https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png"
        )
        self.friend_gps_label = ctk.CTkLabel(self.right_frame, text="Friend GPS: N/A", font=("Consolas", 14), text_color="#32CD32")
        self.friend_gps_label.pack(anchor="w", padx=5, pady=2)

        # ---- Foe Alert ----
        self.foe_label = ctk.CTkLabel(self, text="", font=("Consolas", 18), text_color="#FF0000")
        self.foe_label.grid(row=1, column=0, sticky="ew", pady=5)

        # ---- Model Initialization ----
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("C://Users//Gowtham C K//Desktop//score-streak - Copy//runs//detect//yolo11s//weights//best.pt")

        # ---- Shared frame buffer ----
        self.latest_frame = None
        self.display_frame = None

        # ---- Coordinate storage ----
        self.friend_coords = None
        self.cam_coords = None

        # ---- Networking ----
        self.sock_friend = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_friend.bind(("0.0.0.0", 5005))
        self.sock_cam = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_cam.bind(("0.0.0.0", 5006))

        # ---- Marker State ----
        self.friend_marker = None

        # ---- Threads ----
        threading.Thread(target=self.receive_friend_gps, daemon=True).start()
        threading.Thread(target=self.receive_cam_gps, daemon=True).start()
        threading.Thread(target=self.capture_frames, daemon=True).start()
        threading.Thread(target=self.run_inference, daemon=True).start()
        self.after(30, self.update_ui)

    def receive_friend_gps(self):
        while True:
            data, _ = self.sock_friend.recvfrom(1024)
            lat, lon = map(float, data.decode().split(','))
            self.friend_coords = (lat, lon)
            self.friend_gps_label.configure(text=f"Friend GPS: {lat:.6f}, {lon:.6f}")
            if self.friend_marker:
                self.friend_marker.set_position(lat, lon)
            else:
                self.friend_marker = self.map_widget.set_marker(lat, lon, text="Friend", icon=None)

    def receive_cam_gps(self):
        while True:
            data, _ = self.sock_cam.recvfrom(1024)
            lat, lon = map(float, data.decode().split(','))
            self.cam_coords = (lat, lon)
            self.cam_gps_label.configure(text=f"Camera GPS: {lat:.6f}, {lon:.6f}")

    def capture_frames(self):
        cap = cv2.VideoCapture("udp://0.0.0.0:1234", cv2.CAP_FFMPEG)
        while True:
            ret, frame = cap.read()
            if ret:
                self.latest_frame = frame
            else:
                time.sleep(0.01)

    def run_inference(self):
        while True:
            if self.latest_frame is None:
                time.sleep(0.01)
                continue
            results = self.model.predict(
                self.latest_frame,
                device=self.device,
                imgsz=320,
                half=(self.device=="cuda")
            )
            annotated = results[0].plot()
            self.display_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            # Use stored cam_coords instead of label parsing
            if results[0].boxes and self.cam_coords:
                lat, lon = self.cam_coords
                self.foe_label.configure(text=f"FOE DETECTED at Camera Location: {lat:.6f}, {lon:.6f}")
            else:
                self.foe_label.configure(text="")

    def update_ui(self):
        if self.display_frame is not None:
            img = Image.fromarray(self.display_frame)
            # Fixed size 640x480
            imgtk = CTkImage(light_image=img, size=(640, 480))
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk
        self.after(30, self.update_ui)

if __name__ == "__main__":
    app = MilitaryGPSVisualizer()
    app.mainloop()
