import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Variabel global untuk koordinat crop
x1, y1, x2, y2 = 0, 0, 0, 0
drag_start = False

# Fungsi untuk menangani event mouse
def select_crop(event, x, y, flags, param):
    global x1, y1, x2, y2, drag_start

    if event == cv2.EVENT_LBUTTONDOWN:  # Ketika mouse ditekan, mulai drag
        drag_start = True
        x1, y1 = x, y  # Titik awal crop

    elif event == cv2.EVENT_MOUSEMOVE:  # Ketika mouse bergerak
        if drag_start:  # Jika sedang drag, update titik akhir crop
            x2, y2 = x, y

    elif event == cv2.EVENT_LBUTTONUP:  # Ketika mouse dilepas, akhir drag
        drag_start = False
        x2, y2 = x, y  # Titik akhir crop

# Memuat model YOLOv5
model = YOLO("yolov5s.pt")  # Ganti dengan model YOLOv5 yang diinginkan

# Membaca gambar menggunakan OpenCV
img = cv2.imread('images.jpg')  # Ganti dengan path gambar yang sesuai

# Menampilkan gambar dan menunggu input pengguna untuk memilih area crop
cv2.imshow("Select Crop Area", img)
cv2.setMouseCallback("Select Crop Area", select_crop)

# Tunggu hingga pengguna selesai memilih area crop (menekan 'Enter' untuk konfirmasi)
while True:
    temp_img = img.copy()

    # Gambarkan area crop yang sedang dipilih
    if drag_start or (x2 != 0 and y2 != 0):
        cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Select Crop Area", temp_img)

    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # 'Enter' untuk konfirmasi crop
        break

cv2.destroyAllWindows()

# Crop gambar berdasarkan koordinat yang dipilih
cropped_img = img[y1:y2, x1:x2]

# Mendeteksi objek dalam gambar yang telah di-crop
results_cropped = model(cropped_img)

# Mengonversi gambar dari BGR ke RGB untuk Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

# Render hasil deteksi dengan plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Figure pertama: Menampilkan gambar crop
axes[0].imshow(cropped_img_rgb)
axes[0].set_title("Gambar Crop")
axes[0].axis('off')

# Figure kedua: Menampilkan gambar dengan bounding box yang terdeteksi dari gambar crop
axes[1].imshow(results_cropped[0].plot())  # Gambar dengan bounding box pada gambar yang telah di-crop
axes[1].set_title("Gambar Crop dengan Bounding Box")
axes[1].axis('off')

# Menampilkan kedua gambar
plt.show()
