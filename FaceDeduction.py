import cv2

faceCapture =  cv2.CascadeClassifier("C:/Users/Shubh Raj/AppData/Roaming/Python/Python312/site-packages/cv2/data/haarcascade_frontalface_default.xml")

vc = cv2.VideoCapture(0) 
while True :
    ret , videoData = vc.read()
    if not ret:
        break

    col = cv2.cvtColor(videoData,cv2.COLOR_BGR2GRAY)
    faces = faceCapture.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(videoData,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Video_live",videoData)
    if cv2.waitKey(10) == ord("8"):
        break
vc.release()
cv2.destroyAllWindows()