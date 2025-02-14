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

