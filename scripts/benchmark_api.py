import asyncio
import aiohttp
import time
import cv2
import numpy as np
import io

# Create dummy image
img = np.zeros((224, 224, 3), dtype=np.uint8)
cv2.rectangle(img, (20, 20), (200, 200), (0, 0, 255), -1)
is_success, buffer = cv2.imencode(".jpg", img)
img_bytes = buffer.tobytes()

async def fetch(session, url, data):
    async with session.post(url, data=data) as response:
        return await response.json()

async def main(num_requests=1000, concurrency=50):
    url = "http://127.0.0.1:8000/predict"
    print(f"Starting benchmark: {num_requests} requests with {concurrency} async clients...")
    
    start_time = time.time()
    
    # We use aiohttp FormData to send multipart/form-data
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(num_requests):
            data = aiohttp.FormData()
            data.add_field('file', img_bytes, filename='dummy.jpg', content_type='image/jpeg')
            tasks.append(fetch(session, url, data))
            
            # Simple concurrency control
            if len(tasks) >= concurrency:
                await asyncio.gather(*tasks)
                tasks = []
        
        # Flush remaining
        if tasks:
            await asyncio.gather(*tasks)

    end_time = time.time()
    
    duration = end_time - start_time
    fps = num_requests / duration
    print("=========================================")
    print(f"Benchmark Results (CPU Mode for API):")
    print(f"Total Requests  : {num_requests}")
    print(f"Total Time      : {duration:.2f} seconds")
    print(f"Throughput (RPS): {fps:.2f} req/s")
    print("=========================================")

if __name__ == "__main__":
    asyncio.run(main())
