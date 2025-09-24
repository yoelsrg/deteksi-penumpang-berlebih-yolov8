from ultralytics import YOLO

# Load model YOLOv8 yang sudah dilatih
model = YOLO('yolov8m.pt')  # Ganti dengan model yang sudah dilatih, misalnya yolov8m.pt

# Path gambar untuk pengujian
image_path = 'C:/Users/lenovo/Downloads/yolo/images/gambar.jpg'  # Ganti dengan nama file gambar yang sudah diunduh

# Lakukan prediksi
results = model.predict(image_path)

# Tampilkan hasil deteksi
results[0].show()  # Menggunakan show() pada hasil pertama dalam list

# Jika ingin menyimpan gambar dengan hasil deteksi:
results[0].save()  # Menyimpan gambar dengan deteksi ke file

# cap = cv2.VideoCapture('videos/tes.mp4')