# Facial Recognition-Based Access Control System using Raspberry Pi 5

# MIT ADT University - Loni Kalbhor - Pune (MH)

F.Y. BTech CSE CEW Project Code
Under the Guidance of 
Prof Pranav Chipakatti

## 🧑‍💻 Credits

Created as a CEW Mini-Project using Raspberry Pi, OpenCV, and Python by Group No CEW24252002 and CEW24252003 by the users
1. Pranav Ravindra Vairal [ADT24SOCB1502]
2. Aditya Ashok Zarekar [ADT24SOCB1508]
3. Atharav Bapurao Pawar [ADT24SOCB1514]
4. Ayush Dubey [ADT24SOCB1516]
5. Azmat Ali Siddiqui [ADT24SOCB1517]
6. Nakhate Sarang Pawan [ADT24SOCB1526]
7. Nikam Nikhil Bhaskar [ADT24SOCB1527]
8. Nikhil Rajendra Shelar [ADT24SOCB1528]
9. Rajale Rutuja Sachin [ADT24SOCB1534]
10. Shivani Galinde [ADT24SOCB1538]

## 📌 Project Overview

This project implements a secure and fully offline facial recognition system using Raspberry Pi 5, Camera Module V2, GPIO LEDs, and a 0.96" I2C OLED display.
It recognizes authorized faces using a trained dataset and gives real-time feedback through LEDs and the OLED display. The system starts automatically on boot using `systemd`.

---

## 🧰 Hardware Requirements

- Raspberry Pi 5 (8GB)
- Pi Camera Module V2
- 0.96" OLED Display (SSD1306, I2C)
- Red and Green 5mm LEDs
- 220Ω Resistors (x2)
- FPC Camera Cable
- Breadboard & Jumper Wires
- Optional: Tactile push button

---

## 💻 Software Requirements

- Raspberry Pi OS (Bookworm 64-bit recommended)
- Python 3.9 or above
- Terminal or VNC access

---

## 📦 Installation Instructions

### 🛠 1. Install System Dependencies

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential cmake \
libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev \
libboost-python-dev libboost-thread-dev \
libjpeg-dev libtiff5-dev libjasper-dev libpng-dev \
libavcodec-dev libavformat-dev libswscale-dev \
libatlas-base-dev libqtgui4 python3-pyqt5 \
libqt4-test libilmbase-dev libopenexr-dev \
libgstreamer1.0-dev python3-dev python3-pip
````

> ✅ These libraries are critical for compiling and installing **dlib** and **face\_recognition**.

---

### 🧪 2. Install Python Libraries

```bash
pip3 install numpy opencv-python face_recognition \
imutils adafruit-circuitpython-ssd1306 pillow
```

> ℹ️ If `face_recognition` fails, ensure that `cmake`, `libboost`, and `dlib` are properly installed.

---

### 🔄 3. Enable I2C & Camera

Run:

```bash
sudo raspi-config
```

* Enable I2C under "Interface Options"
* Enable Camera via `libcamera`
* Reboot the Pi

---

## 🧠 Common Errors & Fixes

| **Issue**                                    | **Cause**                           | **Fix**                                       |
| -------------------------------------------- | ----------------------------------- | --------------------------------------------- |
| `dlib installation fails`                    | Missing `cmake` or Boost libraries  | Run `sudo apt install cmake libboost-all-dev` |
| `face_recognition not found`                 | Python path or pip version mismatch | Use `pip3 install face_recognition`           |
| `ModuleNotFoundError: No module named 'cv2'` | OpenCV not installed                | Run `pip3 install opencv-python`              |
| `OSError: libatlas.so not found`             | Missing math libs                   | Install `libatlas-base-dev`                   |
| `OLED not displaying anything`               | Wrong I2C address or wiring         | Confirm address with `sudo i2cdetect -y 1`    |
| `Camera not working`                         | Not enabled or in use               | Run `libcamera-hello` to test                 |

---

## ▶️ Running the Project

### 1. Clone or transfer the code

```bash
cd /home/pi
mkdir facial_recognition && cd facial_recognition
# Copy or clone your code files here
```

### 2. Run manually (for testing)

```bash
python3 main.py
```

### 3. Enable Auto-Start on Boot

```bash
sudo cp facial_recognition.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable facial_recognition.service
sudo systemctl start facial_recognition.service
```

---

## 📸 Add New Users (Manual Method)

1. Run `python3 image_capture.py`
2. Enter name when prompted and capture face images
3. Then run `python3 training.py` to update the model

---

## 📂 File Structure

```
facial_recognition/
├── main.py
├── image_capture.py
├── training.py
├── splash.png
├── facial_recognition.service
└── README.md
```

---

## 🧑‍💻 Credits

Created as a CEW Mini-Project using Raspberry Pi, OpenCV, and Python by Group No CEW24252002 and CEW24252003 by the users
1. Pranav Ravindra Vairal
2. 

```

---

Let me know if you'd like this saved as a `.md` file or copied directly into your repository.
```
