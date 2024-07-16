from flask import Flask, Response, render_template
from ultralytics import YOLO
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('video.html')

def detect_objects():
    # Inisialisasi model YOLO dengan model yang sudah dilatih
    model = YOLO('models/best.pt')
    
    # Mulai capture video dari kamera (indeks 0 untuk kamera default)
    cap = cv2.VideoCapture(0)

    try:
        while True:
            # Ambil frame dari kamera
            ret, frame = cap.read()

            # Jika tidak bisa membaca frame, keluar dari loop
            if not ret:
                break

            # Flip frame secara horizontal (opsional)
            frame = cv2.flip(frame, 1)

            # Deteksi objek menggunakan model YOLO
            results = model(frame)

            # Ambil kotak prediksi, skor, dan kelas dari hasil deteksi
            pred_boxes = results[0].boxes.xyxy.cpu().numpy()
            pred_scores = results[0].boxes.conf.cpu().numpy()
            pred_classes = results[0].boxes.cls.cpu().numpy()

            # Loop melalui semua prediksi dan gambar kotak serta label di atasnya
            for i, box in enumerate(pred_boxes):
                x1, y1, x2, y2 = map(int, box)
                label = f'{model.names[int(pred_classes[i])]} {pred_scores[i]:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Encode frame ke format JPEG
            ret, buffer = cv2.imencode('.jpg', frame)

            # Jika gagal meng-encode frame, lanjutkan ke iterasi berikutnya
            if not ret:
                continue

            # Mengirimkan frame sebagai respons menggunakan teknik multipart/x-mixed-replace
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    finally:
        # Setelah selesai, lepaskan capture video
        cap.release()

@app.route('/video_feed')
def video_feed():
    # Response yang mengirimkan video stream menggunakan generator dari fungsi detect_objects
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Menjalankan aplikasi Flask dalam mode debug
    app.run(debug=True)

# Ini adalah contoh komentar di luar blok kode Flask
