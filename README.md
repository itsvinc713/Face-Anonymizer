// Project: Face Anonymizer
// File: main.py

The project uses mediapipe library for face detection. To modify the project on either image/video/webcam, update lines 40-41 in main.py:
args.add_argument("--mode", default='webcam') # 'webcam' --> 'video' OR 'image'
args.add_argument("--filePath", default=None) # for video or image, you need to change NONE into the directory path.(eg. ./data/video1.png)
