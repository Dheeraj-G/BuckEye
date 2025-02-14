# BuckEye

## The Backend Workflow

1. Capture information from cameras.
2. Place camera feeds into a queue for multithreading.
3. Use Python to access the live feed and pass it into OpenCV for data processing.
4. OpenCV processes the feed:
    - Breaks it into frames.
    - Preprocesses the images.
5. Feed the processed images into the YOLO model.
6. YOLO model detects humans and determines occupancy in zones (seats/tables).
7. Correlate detected zones to tables and seats.
8. Update seat and table information in Supabase.
9. Frontend fetches updated data from the database.
10. Update visuals on the frontend.

## Frontend Notes

As of right now, the frontend is made with Next.js with typescript and tailwind enabled.
Should some of these functionalities not be required, we need to regenerate the frontend.
