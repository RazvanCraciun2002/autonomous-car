import cv2
import socket
from ultralytics import YOLO
import numpy as np
from urllib.request import urlopen
import torch
import time

# === Configurații generale ===
if torch.cuda.is_available():
    print("✅ YOLO rulează pe GPU:", torch.cuda.get_device_name(0))
else:
    print("❌ YOLO rulează pe CPU")

# IP Raspberry Pi
RPI_IP = "192.168.0.158"  # pentru WiFi normal
#RPI_IP = "192.168.238.8"   # pentru hotspot
STREAM_URL = f"http://{RPI_IP}:8080"
SOCKET_PORT = 9999
MODEL_PATH = r"C:\Users\Razvan\Desktop\best.pt"

# === Inițializare Socket & Model ===
sock = socket.socket()
sock.connect((RPI_IP, SOCKET_PORT))
print("Conectat la Raspberry Pi prin socket.")

model = YOLO(MODEL_PATH)
model.to('cuda')

# === Variabile pentru control ===
last_detected_speed = None
last_sent_speed = None
last_sent_time = time.time()
detection_counter = 0
confirmation_threshold = 3

timeout_no_detection = 5  # secunde fără detectare nouă

# === Funcție trimitere comandă ===
def send_command(cmd):
    try:
        sock.send(cmd.encode())
        print(f"✅ Trimis: {cmd} km/h")
    except:
        print("❌ Eroare la trimiterea comenzii!")

# === Stream video de la Raspberry ===
stream = urlopen(STREAM_URL)
bytes_data = b''

while True:
    bytes_data += stream.read(1024)
    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

        results = model(frame, verbose=False)[0]
        annotated = frame.copy()
        send_cmd = None

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # === Filtrare după dimensiunea bounding box-ului ===
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            frame_area = frame.shape[0] * frame.shape[1]

            # Ignoră detecții prea mari (>50% din imagine)
            if box_area > 0.5 * frame_area:
                print(f"❌ Ignorat: bounding box prea mare ({round(100 * box_area/frame_area)}%)")
                continue

            label = model.names[cls]

            if conf >= 0.5:
                # Desenăm bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            try:
                speed_value = int(label)
                print(f"✅ Limită detectată: {speed_value} km/h")

                if speed_value == last_detected_speed:
                    detection_counter += 1
                else:
                    last_detected_speed = speed_value
                    detection_counter = 1

                if detection_counter >= confirmation_threshold:
                    send_cmd = str(speed_value)
                    detection_counter = 0
            except Exception as e:
                print(f"⚠️ Eroare la conversia labelului '{label}' în număr întreg: {e}")


        current_time = time.time()

        if send_cmd:
            if send_cmd != last_sent_speed:
                send_command(send_cmd)
                last_sent_speed = send_cmd
                last_sent_time = current_time
        else:
            # Dacă nu a fost detectat nimic nou, dar a trecut timeout-ul, reconfirmă ultima viteză
            if last_sent_speed and (current_time - last_sent_time > timeout_no_detection):
                send_command(last_sent_speed)
                last_sent_time = current_time
                print(f"🔄 Reconfirmat: păstrăm viteza {last_sent_speed} km/h")

        # Afișăm imaginea adnotată
        cv2.imshow("Detectie Limite Viteză (YOLO)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
sock.close()
