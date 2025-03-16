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

# Load the YOLO model
model = YOLO("yolo11n.pt")

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


"""
Function to process video from a given source (webcam or video file)
Takes a source parameter which can be a webcam index (0) or video file path
Opens video capture, reads frames continuously until video ends
Converts each frame to grayscale for processing
Displays processed frames in a window named after the source
Allows quitting playback by pressing 'q'
Cleans up by releasing capture and closing windows when done
"""
def process_video(source):
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # It seems that yolo is trained on rgb images so the use of grayscaled images would not be useful
        
        bounded, output = yolo_process(frame) # Process the frame through yolov11
        
        cv2.imshow(f"Video {source}", bounded) # Display the results
        
        print(output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Main function to perform the required tasks
if __name__ == "__main__":
    sources = [0]  # List of video sources to process (The default is zero which will attempt to use the computer webcam)
    processes = []

    # Processes the information in each stream
    for src in sources:
        p = mp.Process(target=process_video, args=(src,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



# Process the video