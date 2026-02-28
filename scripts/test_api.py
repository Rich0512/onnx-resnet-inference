import requests
import cv2
import numpy as np
import io

# Create dummy image
img = np.zeros((224, 224, 3), dtype=np.uint8)
cv2.rectangle(img, (20, 20), (200, 200), (0, 0, 255), -1)
is_success, buffer = cv2.imencode(".jpg", img)
io_buf = io.BytesIO(buffer)

print("Sending POST request to http://127.0.0.1:8000/predict ...")
files = {'file': ('dummy.jpg', io_buf, 'image/jpeg')}
try:
    response = requests.post("http://127.0.0.1:8000/predict", files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.json()}")
except Exception as e:
    print(f"Failed to connect: {e}")
