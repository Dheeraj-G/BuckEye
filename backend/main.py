"""
This is the main file for the backend of the application
It will take the video data from the cameras
Multithread the videos so they run almost simultaneously
The video information will be inserted into OpenCV
OpenCV will be used to preprocess the videos into images
The images will be put into a the yolov11 model to detect people in zones
Correlate the detected zones into tables and seats
Information is then reflected in supabase using a websocket
Websocket is then used to update the frontend
"""

# Import the necessary libraries
# use of multiprocessing instead of threading to avoid GIL
from ultralytics import YOLO
import cv2
import multiprocessing as mp
import time
import asyncio
import websockets
import json
from multiprocessing import Queue

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Create a queue for inter-process communication
message_queue = Queue()

# Calculate Intersection over Union (IoU) between two boxes.
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    # Calculate intersection area
    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)
    
    # If there is no intersection
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # Calculate union area
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)
    union_area = box1_area + box2_area - inter_area
    
    # Return IoU
    return inter_area / union_area

# Check if tables are filled based on people overlap and return an array of 0s and 1s.
def check_table_filled(tables, people, iou_threshold):
    filled = []
    
    # Iterate over each table
    for table_idx, table in enumerate(tables):
        table_filled = 0  # Start by assuming the table is empty
        
        # Check if any person overlaps with this table
        for person in people:
            if iou(table, person) > iou_threshold:
                table_filled = 1  # If overlap is above threshold, mark as filled
                break  # No need to check other people once the table is filled
        
        filled.append(table_filled)  # Append the result for this table
    
    return filled

def yolo_process(frame):
    results = model(frame)
    tables = set()
    people = set()
    
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]  # Class label
            
            if label.lower() == "dining table":
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                
                # Draw bounding box (This part is simply for our visualization)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Get the table area information
                tables.add((x1-30, x2+30, y1-30, y2+30))
            
            if label.lower() == "person": # Only perform the actions if it is a person
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                
                # Draw bounding box (This part is simply for our visualization)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                people.add((x1,x2,y1,y2))
        
    # check the bounding box against the seat bounding box coordinates
    output = check_table_filled(tables, people, 0)
    
    return (frame, output)

def process_video(source, queue):
    """Process video from a single source and put results in the queue"""
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        bounded, output = yolo_process(frame)
        
        # Put data in queue for websocket transmission
        data = {
            'timestamp': time.time(),
            'source': source,
            'table_status': output
        }
        queue.put(data)
        
        cv2.imshow(f"Video {source}", bounded)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

async def websocket_sender(websocket, queue):
    """Continuously send data from queue through websocket"""
    try:
        while True:
            if not queue.empty():
                data = queue.get()
                try:
                    await websocket.send(json.dumps(data))
                except websockets.exceptions.ConnectionClosed:
                    print("Websocket connection closed")
                    break
            await asyncio.sleep(0.01)  # Small delay to prevent CPU overuse
    except Exception as e:
        print(f"Error in websocket sender: {e}")

async def handle_connection(websocket):
    """Handle a single websocket connection"""
    try:
        # Send initial connection message
        await websocket.send(json.dumps({"status": "connected"}))
        
        # Start video processing in separate processes
        processes = []
        sources = [0, 1]  # List of video sources
        
        for src in sources:
            p = mp.Process(target=process_video, args=(src, message_queue))
            p.start()
            processes.append(p)
        
        # Start websocket sender
        await websocket_sender(websocket, message_queue)
        
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Cleanup processes
        for p in processes:
            p.terminate()
            p.join()

async def main():
    """Main function to start websocket server"""
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("WebSocket server started at ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
        
    # One websocket to be open and continually send updated data into the database
    # in that case the different threads may need to be included in the websocket function
