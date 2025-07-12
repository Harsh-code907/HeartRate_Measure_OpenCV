import cv2
import numpy as np
import mediapipe as mp
import time
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
from collections import deque
import threading

# To filter noise using butterworth filter
def bandpass_filter(signal, fs=30, low=0.75, high=3.0, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

# Plot graph simaltenously with video
plot_window = 300  # ~10 seconds
signal_buffer = deque(maxlen=plot_window)

def live_plot():
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'r')
    ax.set_ylim(-2, 2)
    ax.set_xlim(0, plot_window)
    ax.set_title("Filtered Heartbeat Signal")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Amplitude")

    while True:
        if len(signal_buffer) > 0:
            y = list(signal_buffer)
            x = list(range(len(y)))
            line.set_data(x, y)
            ax.set_xlim(0, len(y))
            ax.relim()
            ax.autoscale_view(True, True, True)
            fig.canvas.draw()
            fig.canvas.flush_events()
        time.sleep(0.03)

# Threading
threading.Thread(target=live_plot, daemon=True).start()

#Forehead detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

green_values = []
timestamps = []
bpm_history = []

start_time = time.time()
bpm_display = "Calculating..."
bpm_final = None
last_bpm_update = 0
update_interval = 5
calculation_duration = 10

heart_visible = False
last_heartbeat_time = 0
HEART_DISPLAY_DURATION = 0.2 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)
    frame_display = frame.copy()

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        fx = int(landmarks[10].x * w)
        fy = int(landmarks[10].y * h) - 20
        roi_size = 30
        x1 = max(fx - roi_size, 0)
        y1 = max(fy - roi_size, 0)
        x2 = min(fx + roi_size, w)
        y2 = min(fy + roi_size, h)

        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            green = np.mean(roi[:, :, 1])
            green_values.append(green)
            timestamps.append(time.time() - start_time)
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (255, 0, 255), 4)

    if len(timestamps) > plot_window:
        green_values = green_values[-plot_window:]
        timestamps = timestamps[-plot_window:]

    current_time = time.time()
    if (current_time - start_time > calculation_duration and
        current_time - last_bpm_update > update_interval and
        len(green_values) > 100):

        fs = len(green_values) / (timestamps[-1] - timestamps[0])
        signal = np.array(green_values)
        signal = (signal - np.mean(signal)) / np.std(signal)
        filtered = bandpass_filter(signal, fs=fs)

        signal_buffer.clear()
        signal_buffer.extend(filtered.tolist())

        peaks, _ = find_peaks(filtered, distance=fs/2)
        if len(peaks) > 1:
            peak_times = [timestamps[p] for p in peaks]
            intervals = np.diff(peak_times)
            if len(intervals) > 0:
                avg_interval = np.mean(intervals)
                bpm = 60 / avg_interval
                if 45 < bpm < 180:
                    bpm_history.append(bpm)
                    if len(bpm_history) > 5:
                        bpm_history = bpm_history[-5:]
                    bpm_final = int(np.mean(bpm_history))
                    bpm_display = f"Your heart rate is {bpm_final} BPM"

                    # Trigger blinking heart on new peaks
                    for peak_time in peak_times:
                        if peak_time > last_bpm_update:
                            heart_visible = True
                            last_heartbeat_time = time.time()

        last_bpm_update = current_time


    if heart_visible:
        cv2.putText(frame_display, "❤️", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    if time.time() - last_heartbeat_time > HEART_DISPLAY_DURATION:
        heart_visible = False

    # Show BPM
    cv2.putText(frame_display, bpm_display, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    cv2.imshow("Heart Rate Monitor", frame_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
