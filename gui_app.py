import tkinter as tk
from tkinter import filedialog, simpledialog
from ultralytics import YOLO
import cv2
import threading
import numpy as np
import mss
from streamlink import Streamlink

# Load model YOLO
model = YOLO("yolov8s.pt")

# Warna kotak
colors = {
    "person": (0, 255, 0),
    "motorcycle": (255, 255, 0),
    "overload": (0, 0, 255)
}

def draw_label(img, text, x, y, color):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y - 20), (x + w, y), color, -1)
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def deteksi_frame(frame):
    results = model(frame)[0]
    persons, motorcycles = [], []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if label == "person":
            persons.append((x1, y1, x2, y2))
        elif label == "motorcycle":
            motorcycles.append((x1, y1, x2, y2))

    for motor_box in motorcycles:
        mx1, my1, mx2, my2 = motor_box
        penumpang = 0
        terdeteksi = []

        for px1, py1, px2, py2 in persons:
            if px1 > mx1 - 50 and px2 < mx2 + 50 and py2 < my2 + 100:
                penumpang += 1
                terdeteksi.append((px1, py1, px2, py2))

        cv2.rectangle(frame, (mx1, my1), (mx2, my2), colors["motorcycle"], 2)
        draw_label(frame, f"Sepeda Motor - {penumpang} penumpang", mx1, my1, colors["motorcycle"])

        for px1, py1, px2, py2 in terdeteksi:
            cv2.rectangle(frame, (px1, py1), (px2, py2), colors["person"], 2)
            draw_label(frame, "Orang", px1, py1, colors["person"])

        if penumpang > 2:
            draw_label(frame, "Penumpang Berlebih", mx1, my2 + 35, colors["overload"])
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), colors["overload"], 2)

    cv2.imshow("Deteksi Pelanggaran", frame)

def proses_video(source):
    cap = cv2.VideoCapture(source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        deteksi_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

def proses_layar(area):
    sct = mss.mss()
    while True:
        img = np.array(sct.grab(area))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        deteksi_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

def pilih_video():
    path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
    if path:
        threading.Thread(target=proses_video, args=(path,), daemon=True).start()

def mulai_webcam():
    threading.Thread(target=proses_video, args=(0,), daemon=True).start()

class ScreenSelector(tk.Toplevel):
    def __init__(self, master, callback):
        super().__init__(master)
        self.callback = callback
        self.attributes("-fullscreen", True)
        self.attributes("-alpha", 0.3)
        self.config(bg="black")
        self.canvas = tk.Canvas(self, cursor="cross", bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.start_x = self.start_y = None
        self.rect = None
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_press(self, event):
        self.start_x = self.winfo_pointerx()
        self.start_y = self.winfo_pointery()
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)

    def on_drag(self, event):
        x, y = self.winfo_pointerx(), self.winfo_pointery()
        self.canvas.coords(self.rect, self.start_x, self.start_y, x, y)

    def on_release(self, event):
        end_x = self.winfo_pointerx()
        end_y = self.winfo_pointery()
        left = min(self.start_x, end_x)
        top = min(self.start_y, end_y)
        width = abs(self.start_x - end_x)
        height = abs(self.start_y - end_y)
        self.destroy()
        self.callback({"top": top, "left": left, "width": width, "height": height})

def mulai_pilih_area():
    def mulai(area):
        threading.Thread(target=proses_layar, args=(area,), daemon=True).start()
    ScreenSelector(app, mulai)

def mulai_streamlink():
    url = simpledialog.askstring("Streaming URL", "Masukkan URL streaming (YouTube/RTMP):")
    if url and url.startswith(("http", "rtmp")):
        def jalankan():
            try:
                session = Streamlink()
                streams = session.streams(url)
                if 'best' in streams:
                    stream_url = streams['best'].url
                    threading.Thread(target=proses_video, args=(stream_url,), daemon=True).start()
                else:
                    print("Stream 'best' tidak ditemukan.")
            except Exception as e:
                print(f"Gagal mengambil stream: {e}")
        threading.Thread(target=jalankan, daemon=True).start()
    else:
        print("URL tidak valid.")

# -------- GUI --------

app = tk.Tk()
app.title("Deteksi Penumpang Berlebih")
app.geometry("420x450")
app.config(bg="#f0f2f5")

# Frame utama
frame = tk.Frame(app, bg="#f0f2f5")
frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

title = tk.Label(frame, text="Deteksi Penumpang Sepeda Motor", font=("Helvetica", 15, "bold"), bg="#f0f2f5", fg="#333")
title.pack(pady=(0, 25))

btn_params = {
    "font": ("Helvetica", 14),
    "width": 25,
    "height": 2,
    "bd": 0,
    "relief": tk.FLAT,
    "cursor": "hand2"
}

# Warna tombol
btn_colors = {
    "video": "#1976d2",
    "webcam": "#388e3c",
    "screen": "#7b1fa2",
    "stream": "#f4511e"
}

def on_enter(e):
    e.widget['background'] = e.widget.hover_bg

def on_leave(e):
    e.widget['background'] = e.widget.default_bg

# Tombol video
btn_video = tk.Button(frame, text="📂 Pilih Video", bg=btn_colors["video"], fg="white", **btn_params, command=pilih_video)
btn_video.default_bg = btn_colors["video"]
btn_video.hover_bg = "#115293"
btn_video.bind("<Enter>", on_enter)
btn_video.bind("<Leave>", on_leave)
btn_video.pack(pady=8)

# Tombol webcam
btn_webcam = tk.Button(frame, text="📹 Webcam", bg=btn_colors["webcam"], fg="white", **btn_params, command=mulai_webcam)
btn_webcam.default_bg = btn_colors["webcam"]
btn_webcam.hover_bg = "#27632a"
btn_webcam.bind("<Enter>", on_enter)
btn_webcam.bind("<Leave>", on_leave)
btn_webcam.pack(pady=8)

# Tombol area layar
btn_screen = tk.Button(frame, text="📺 Area Layar", bg=btn_colors["screen"], fg="white", **btn_params, command=mulai_pilih_area)
btn_screen.default_bg = btn_colors["screen"]
btn_screen.hover_bg = "#4a148c"
btn_screen.bind("<Enter>", on_enter)
btn_screen.bind("<Leave>", on_leave)
btn_screen.pack(pady=8)

# Tombol streaming
btn_stream = tk.Button(frame, text="🌐 Streaming (YouTube/RTMP)", bg=btn_colors["stream"], fg="white", **btn_params, command=mulai_streamlink)
btn_stream.default_bg = btn_colors["stream"]
btn_stream.hover_bg = "#b23a00"
btn_stream.bind("<Enter>", on_enter)
btn_stream.bind("<Leave>", on_leave)
btn_stream.pack(pady=8)

# Footer label
footer = tk.Label(app, text="Tekan tombol 'q' pada jendela deteksi untuk keluar.", font=("Helvetica", 10), bg="#f0f2f5", fg="#666")
footer.pack(pady=12)

app.mainloop()
