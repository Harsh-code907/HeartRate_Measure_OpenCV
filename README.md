🫀 Project: Contactless Heart Rate Detection Using a Webcam

**Medical Term:** *Remote Photoplethysmography (rPPG)*

📌 **Overview:**
This project demonstrates how we can estimate a person's **heart rate** using only a **webcam and computer vision**—no physical sensors required. The process is based on a medical technique called **remote photoplethysmography (rPPG)**.

---

### 🧠 How It Works (Medically & Technically):

💡 **Photoplethysmography (PPG)** is the medical technique of measuring blood volume changes through light absorption—traditionally done using fingertip or wrist sensors.

📸 **Remote PPG (rPPG)** achieves the same non-invasively by analyzing **color fluctuations** on the skin (typically the face) due to blood flow. Here's what happens step by step:

1. **Face Detection:**
   Using **MediaPipe**, we detect facial landmarks to localize a stable skin region (like the forehead).

2. **Color Signal Extraction:**
   We monitor subtle **green color variations** in the skin over time. Green is most responsive to changes in blood flow due to oxygenated hemoglobin's light absorption properties.

3. **Signal Processing:**
   We apply a **bandpass filter** to isolate frequency components between **0.75–3.0 Hz** (i.e., 45–180 BPM range).

4. **Heartbeat Detection:**
   Peaks in the processed signal represent heartbeats. The time between peaks gives the **instantaneous BPM (beats per minute)**.

5. **Visualization:**

   * Real-time graph of the heartbeat signal
   * On-screen BPM reading
   * Animated heart icon to represent detected beats

---

### 📊 What Is a Normal Heart Rate?

Heart rate depends on **age, fitness, and activity level**. Here are general guidelines:

| **State**                | **Normal BPM** |
| ------------------------ | -------------- |
| Resting (Adult)          | 60–100         |
| Athlete (Resting)        | 40–60          |
| During exercise          | 120–180        |
| After heavy exertion     | 100–160        |
| During sleep             | 40–70          |
| Stress or anxiety        | ↑ increases    |
| Immediately after eating | Slight ↑       |

🧘‍♂️ In this test, the user is calm and seated, so a reading of **64–72 BPM** is expected and healthy.

---

### 🛠️ Technologies Used:

* **Python**
* **OpenCV** (video capture & display)
* **MediaPipe FaceMesh** (facial landmark tracking)
* **NumPy & SciPy** (signal processing)
* **Matplotlib** (real-time graph)

---

### ✅ Applications:

* Wellness apps
* Contactless health monitoring
* Fitness tracking
* Telemedicine & research

