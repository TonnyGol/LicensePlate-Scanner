# 🚔 LicensePlate Scanner — Automatic Number Plate Recognition

A real-time stolen vehicle detection system built with Python. Uses a webcam to detect license plates via TensorFlow Lite, checks them against a police database, and displays alerts through a modern web dashboard.

## Architecture

```
┌─────────────┐      UDP       ┌──────────────┐      UDP       ┌────────────────┐
│  Web Client  │◄────────────►│  PS-Detector   │◄────────────►│ Server+DataBase │
│  (Flask UI)  │   port 8000   │ (Proxy + OCR)  │   port 8900   │  (SQLite + UDP) │
└─────────────┘               └──────────────┘               └────────────────┘
```

| Component | Role |
|---|---|
| **Server+DataBase** | SQLite database with officer credentials & stolen plate numbers. Validates logins and plate lookups via UDP. |
| **PS-Detector** | Opens webcam, detects plates with TFLite + Tesseract OCR, verifies against the server, and acts as a network proxy between the Web Client and Server. |
| **Web Client** | Flask web app with a glassmorphism UI. Officers log in and monitor a live dashboard that shows stolen plate alerts. |

## How It Works

1. **Officer logs in** via the Web Client → credentials are forwarded through the PS-Detector proxy → validated by the Server against the `Cops` table.
2. **PS-Detector scans** the webcam for license plates using a TFLite object detection model + Tesseract OCR.
3. When a plate is detected, it's **sent to the Server** which checks the `Cars` table.
4. If the plate is in the database → the Web Client dashboard shows a **"STOLEN"** warning with a red pulsing animation for 15 seconds.

## Setup & Run

### Prerequisites
- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed
- Webcam connected

### Install Dependencies
```bash
pip install flask opencv-python numpy pytesseract tflite-runtime
```

### Run (3 terminals)

```bash
# Terminal 1 — Server
cd Server+DataBase
python server_DB.py

# Terminal 2 — Detector + Proxy
cd PS-Detector
python TFLite_detection_webcam.py

# Terminal 3 — Web Client
cd "Web Client"
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Database

The SQLite database (`Police_DB.db`) has two tables:

| Table | Columns | Purpose |
|---|---|---|
| `Cops` | `username`, `password` | Officer login credentials |
| `Cars` | `num` | Stolen license plate numbers |