from cv2 import cv2
import time
import mediapipe as mp

cam = cv2.VideoCapture(0)
prev_time = 0


#####
Hand_mp = mp.solutions.hands
Hand_detect = Hand_mp.Hands()
mpDraw = mp.solutions.drawing_utils

while True :
    suc , frame = cam.read()
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    detect = Hand_detect.process(img_rgb)

    if detect.multi_hand_landmarks :
        for hand in detect.multi_hand_landmarks :
            mpDraw.draw_landmarks(frame,hand,Hand_mp.HAND_CONNECTIONS)
            for id, lm in enumerate(hand.landmark):
                fh , fw , fc = frame.shape
                x,y  = int(fw*lm.x) , int(fh*lm.y)
                cv2.putText(frame,f'{id}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(200,0,0),1)
                #cv2.circle(frame,(x,y),2,(0,100,0) , 1)

    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time

    cv2.putText(frame,f'FPS: {int(fps)}',(30,30),cv2.FONT_HERSHEY_DUPLEX ,1 ,(0,200,0) , 1)

    cv2.imshow("Webcam" , frame)
    if cv2.waitKey(1) & 0xFF ==ord('q') :
        break

cam.release()
cv2.destroyAllWindows()