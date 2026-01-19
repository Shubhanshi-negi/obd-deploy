from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import cv2
import base64
import numpy as np
import os

app = FastAPI()

model = YOLO("yolov8n.pt")

@app.get("/")
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_text()

        img_bytes = base64.b64decode(data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = model(frame, conf=0.4)

        detections = []
        for r in results:
            for b in r.boxes:
                detections.append({
                    "x1": int(b.xyxy[0][0]),
                    "y1": int(b.xyxy[0][1]),
                    "x2": int(b.xyxy[0][2]),
                    "y2": int(b.xyxy[0][3]),
                    "label": model.names[int(b.cls)]
                })

        await ws.send_json(detections)
