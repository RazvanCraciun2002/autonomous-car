import cv2
import numpy as np
import socket
import threading
from picarx import Picarx
from time import sleep
from http.server import BaseHTTPRequestHandler, HTTPServer
from picamera2 import Picamera2
import time

px = Picarx()
sleep(0.2)

prev_steering_angle = 0
alpha = 0.2
debug = True
px.set_cam_pan_angle(20)
px.set_cam_tilt_angle(10)

target_speed = 90  # Viteza default 90 km/h
last_cmd_time = time.time()

ultrasonic_distance = 999
adjusted_speed = 0.8
consecutive_low_readings = 0
LOW_DISTANCE_THRESHOLD = 15
MAX_LOW_READINGS = 3
in_stop = False

MIN_SPEED = 20
MAX_SPEED = 90

last_speed = 0
speed_alpha = 0.2  # coeficient pentru accelerare progresivă (mai mic = mai lent)


def read_ultrasonic():
    global ultrasonic_distance, adjusted_speed, consecutive_low_readings, in_stop
    SAFE_DISTANCE_THRESHOLD = 30
    SAFE_CONFIRM_READINGS = 5
    safe_counter = 0

    while True:
        distance = round(px.ultrasonic.read(), 2)
        if distance <= 0:
            distance = 999
        ultrasonic_distance = distance

        base_speed = 0.1 + ((target_speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)) * (20.0 - 0.1)
        base_speed = round(base_speed, 2)

        if distance < LOW_DISTANCE_THRESHOLD:
            consecutive_low_readings += 1
        else:
            consecutive_low_readings = 0

        if consecutive_low_readings >= MAX_LOW_READINGS:
            adjusted_speed = 0
            in_stop = True
            safe_counter = 0  # Resetăm contorul de siguranță
        elif in_stop:
            if distance > SAFE_DISTANCE_THRESHOLD:
                safe_counter += 1
                if safe_counter >= SAFE_CONFIRM_READINGS:
                    in_stop = False
                    safe_counter = 0
            else:
                safe_counter = 0
            adjusted_speed = 0
        elif LOW_DISTANCE_THRESHOLD <= distance <= 30:
            factor = (distance - LOW_DISTANCE_THRESHOLD) / (30 - LOW_DISTANCE_THRESHOLD)
            adjusted_speed = round(0.1 + factor * (base_speed - 0.1), 2)
        elif distance > 30:
            adjusted_speed = base_speed


        time.sleep(0.05)

def image_preprocessor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask_white = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    height = mask_white.shape[0]
    start_row = int(height * 0.7)
    mask_cropped = np.zeros_like(mask_white)
    mask_cropped[start_row:, :] = mask_white[start_row:, :]
    return mask_cropped

def detect_lanes(frame, mask):
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=100)
    left, right = [], []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.3:
                    continue
                (left if slope < 0 else right).append((x1, y1, x2, y2))
    return avg_line(left, frame), avg_line(right, frame)

def avg_line(lines, frame):
    if not lines:
        return None
    x, y = [], []
    for x1, y1, x2, y2 in lines:
        x += [x1, x2]
        y += [y1, y2]
    poly = np.polyfit(y, x, 1)
    y1, y2 = frame.shape[0], int(frame.shape[0] * 0.6)
    return (int(np.polyval(poly, y1)), y1, int(np.polyval(poly, y2)), y2)

def guide_robot(left, right):
    global prev_steering_angle, last_speed

    angle = 0
    if left:
        error = left[0] - 30
        angle = int(0.1 * error)
    elif right:
        error = right[0] - 610
        angle = int(0.1 * error)

    filtered = int(alpha * angle + (1 - alpha) * prev_steering_angle)
    prev_steering_angle = filtered
    filtered = np.clip(filtered, -25, 25)
    px.set_dir_servo_angle(filtered)

    # Reducere viteză la unghi mare de viraj
    if abs(filtered) > 15:
        speed_factor = 0.5  # Redu la 60% din viteză când virează tare
    else:
        speed_factor = 1.0

    if adjusted_speed == 0:
        print("🚫 STOP: prea aproape (confirmat)")
        px.stop()
        last_speed = 0
    else:
        target = adjusted_speed * speed_factor
        last_speed = speed_alpha * target + (1 - speed_alpha) * last_speed
        last_speed = round(min(last_speed, target), 2)
        print(f"✅ Viteză aplicată: {last_speed:.2f} (filtrat, viraj={filtered}°)")
        px.forward(last_speed)



def socket_listener():
    global target_speed, last_cmd_time
    s = socket.socket()
    s.bind(('0.0.0.0', 9999))
    s.listen(1)
    conn, _ = s.accept()
    print("Client conectat la socket!")

    while True:
        try:
            cmd = conn.recv(1024).decode()
            if cmd:
                last_cmd_time = time.time()
                if cmd.isdigit():
                    speed_value = int(cmd)
                    if MIN_SPEED <= speed_value <= MAX_SPEED:
                        target_speed = speed_value
                        print(f"🚗 Viteză setată: {target_speed} km/h")
        except:
            px.stop()
            break

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/':
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()

        while True:
            frame = picam2.capture_array("main")
            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
            self.wfile.write(b"--frame\r\n")
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg.tobytes())
            self.wfile.write(b"\r\n")
            time.sleep(0.033)

def start_stream():
    global picam2
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        controls={"FrameDurationLimits": (33333, 33333), "AwbMode": 1}
    )
    picam2.configure(video_config)
    picam2.start()
    sleep(1)
    server = HTTPServer(('0.0.0.0', 8080), StreamHandler)
    print("MJPEG stream pe portul 8080")
    server.serve_forever()

def lane_follower():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 320)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        mask = image_preprocessor(frame)
        left_avg, right_avg = detect_lanes(frame, mask)
        guide_robot(left_avg, right_avg)

        if debug:
            result = frame.copy()
            if left_avg:
                cv2.line(result, (left_avg[0], left_avg[1]), (left_avg[2], left_avg[3]), (255, 0, 0), 4)
            if right_avg:
                cv2.line(result, (right_avg[0], right_avg[1]), (right_avg[2], right_avg[3]), (255, 0, 0), 4)
            cv2.imshow("Procesare Linii - RPi", result)
            cv2.imshow("Masca Alb-Negru", mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                px.stop()
                cap.release()
                cv2.destroyAllWindows()
                break

# === Rulează firele ===
t1 = threading.Thread(target=start_stream)
t2 = threading.Thread(target=socket_listener)
t3 = threading.Thread(target=lane_follower)
t4 = threading.Thread(target=read_ultrasonic)

t1.start()
t2.start()
t3.start()
t4.start()
