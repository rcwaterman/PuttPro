import cv2

try:
    cam = cv2.VideoCapture(0)
except:
    print("Error establishing camera connection...")
    exit()

while 1: 
    ret, frame = cam.read()
    cv2.imshow("Video Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    

