from ultralytics import YOLO
import cv2

# Load model YOLOv8
model = YOLO("yolov8n.pt")

# Buka video
cap = cv2.VideoCapture("videos/tes.mp4")

# Warna untuk label
colors = {
    "person": (0, 255, 0),         # Hijau
    "motorcycle": (255, 255, 0),   # Kuning
    "overload": (0, 0, 255)        # Merah
}

def draw_label(img, text, x, y, color):
    """Gambar label dengan background semi-transparan"""
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y - 20), (x + w, y), color, -1)
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    persons = []
    motorcycles = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "person":
            persons.append((x1, y1, x2, y2))
        elif label == "motorcycle":
            motorcycles.append((x1, y1, x2, y2))

    # Proses tiap motor
    for motor_box in motorcycles:
        mx1, my1, mx2, my2 = motor_box
        penumpang = 0
        penumpang_terdeteksi = []

        for person_box in persons:
            px1, py1, px2, py2 = person_box

            if px1 > mx1 - 50 and px2 < mx2 + 50 and py2 < my2 + 100:
                penumpang += 1
                penumpang_terdeteksi.append(person_box)

        # Gambar motor
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), colors["motorcycle"], 2)
        draw_label(frame, f"Motor - {penumpang} penumpang", mx1, my1, colors["motorcycle"])

        # Tampilkan bounding box untuk penumpang di motor
        for px1, py1, px2, py2 in penumpang_terdeteksi:
            cv2.rectangle(frame, (px1, py1), (px2, py2), colors["person"], 2)
            draw_label(frame, "Penumpang", px1, py1, colors["person"])

        # Tandai pelanggaran
        if penumpang > 2:
            draw_label(frame, "Penumpang Berlebih", mx1, my2 + 35, colors["overload"])
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), colors["overload"], 2)

    cv2.imshow("Deteksi Pelanggaran Sepeda Motor", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
