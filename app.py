from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import cv2, base64, numpy as np

app = FastAPI()
model = YOLO("yolov8n.pt")

@app.get("/")
def home():
    return HTMLResponse(open("index.html").read())

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_text()
        img = base64.b64decode(data)
        frame = cv2.imdecode(np.frombuffer(img, np.uint8), 1)

        results = model(frame)
        boxes = []

        for r in results:
            for b in r.boxes:
                boxes.append({
                    "x1": int(b.xyxy[0][0]),
                    "y1": int(b.xyxy[0][1]),
                    "x2": int(b.xyxy[0][2]),
                    "y2": int(b.xyxy[0][3]),
                    "cls": model.names[int(b.cls)]
                })

        await ws.send_json(boxes)
