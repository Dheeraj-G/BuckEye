# This is the main file for the backend of the application
# It will take the video data from the cameras
# Insert the videos into a queue
# Multithread the videos so they run almost simultaneously
# The video information will be inserted into OpenCV
# OpenCV will be used to preprocess the videos into images
# The images will be put into a the yolov11 model to detect people in zones
# Correlate the detected zones into tables and seats
# Information is then reflected in supabase using a websocket
# Websocket is then used to update the frontend

# Import the necessary libraries
# use of multiprocessing instead of threading to avoid GIL
from ultralytics import YOLO
import cv2
import multiprocessing as mp
import time
from queue import Queue

# gets the live video data from phone cameras
def get_video():
    #TODO: get the live video data from phone cameras
    pass



# Function to process video from a given source (webcam or video file)
# Takes a source parameter which can be a webcam index (0) or video file path
# Opens video capture, reads frames continuously until video ends
# Converts each frame to grayscale for processing
# Displays processed frames in a window named after the source
# Allows quitting playback by pressing 'q'
# Cleans up by releasing capture and closing windows when done
def process_video(source):
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Process the frame (e.g., grayscale conversion)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow(f"Video {source}", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Main function to perform the required tasks
if __name__ == "__main__":
    sources = []  # List of video sources to process
    processes = []

    for src in sources:
        p = mp.Process(target=process_video, args=(src,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()













# Load the YOLO model
model = YOLO("yolov11n.pt")


# Process the video