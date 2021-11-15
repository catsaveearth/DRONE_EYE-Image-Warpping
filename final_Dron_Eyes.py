from tkinter import *
from PIL import ImageTk, Image
import cv2
import dlib
import numpy as np
import math
from math import dist
from utils import *
import keyboard, time
from threading import Thread
from skimage import exposure


w, h = 640, 480

#global variable
font = cv2.FONT_ITALIC
frame = None

horizon = False
retrial_brow = True
brow_basis = 0

threshold_value = 30
eye_detect = False
i = 0
time_value = 0
case_cnt = list([0, 0, 0, 0, 0, 0]) #None, left, right, rot_ccw, rot_cw, close

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_lefteye_2splits.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# 눈 상태들
leftState = None #None, left, right, close
rightState = None
blinkCnt = 0
blinkTimer = 0
pre_rightState = None
pre_leftState = None


#드론 상태
flying = False
showStream = 0
tello = None
tello_frame_read = None

# 상태 변화 check -> 드론 움직임
def selectMotion():
    global leftState, rightState, case_cnt, blinkCnt, flying, tello

    if leftState == "None" and rightState == "None": #None 상태가 오랫동안 지속되면 초기화
        case_cnt[0] += 1
        if case_cnt[0] == 2:
            case_cnt = list([0, 0, 0, 0, 0, 0]) #None, left, right, rot_ccw, rot_cw, close
            Current_motion.config(text="No motion")

    elif leftState == "left" and rightState == "left":
        case_cnt[1] += 1

        if case_cnt[1] >= 2:
            if flying == True:
                tello.move_left(20)
                print("move_left")
            Current_motion.config(text="move_left")
            case_cnt = list([0, 0, 0, 0, 0, 0])

    elif leftState == "right" and rightState == "right":
        case_cnt[2] += 1

        if case_cnt[2] >= 2:
            if flying == True:
                tello.move_right(20)
                print("move_right")
            Current_motion.config(text="move_right")
            case_cnt = list([0, 0, 0, 0, 0, 0])

    elif leftState == "close" and rightState != "close":
        case_cnt[3] += 1

        if case_cnt[3] >= 2:
            if flying == True:
                tello.rotate_counter_clockwise(30)
                print("rotate CCW")
            Current_motion.config(text="rotate CCW")
            case_cnt = list([0, 0, 0, 0, 0, 0])

    elif leftState != "close" and rightState == "close":
        case_cnt[4] += 1

        if case_cnt[4] >= 2:
            if flying == True:
                tello.rotate_clockwise(20)
                print("rotate CW")
            Current_motion.config(text="rotate CW")
            case_cnt = list([0, 0, 0, 0, 0, 0])

    elif leftState == "close" and rightState == "close":
        case_cnt[5] += 1

        if case_cnt[5] >= 3:
            if flying == False:
                tello.takeoff()
                flying = True
                print("takeoff")
                Current_motion.config(text="takeoff")
            else:
                tello.land()
                flying = False
                print("land")
                Current_motion.config(text="land")
            case_cnt = list([0, 0, 0, 0, 0, 0])

def checkChangEyes():
    global leftState, rightState, blinkCnt, pre_rightState, pre_leftState, blinkTimer, tello

    if time_value - blinkTimer > 3:
        if blinkCnt == 2:
            blinkCnt = 0
            if flying == True:
                tello.move_forward(30)
                print("move_front")
            Current_motion.config(text="move_front")


        elif blinkCnt == 3:
            blinkCnt = 0
            if flying == True:
                tello.move_back(30)
                print("move_back")
            Current_motion.config(text="move_back")

        else:
            blinkCnt = 0

    if pre_leftState == "close" and leftState != "close":
        if pre_rightState == "close" and rightState != "close":
            if blinkCnt == 0:
                blinkTimer = time_value
            blinkCnt += 1

    pre_leftState = leftState
    pre_rightState = rightState


# face detection
def face_landmark(gray, frame):
    global eye_detect, horizon, brow_basis, leftState, rightState

    if eye_detect == False:
        return

    # dlib (face_landmark)
    rects = detector(gray, 1) # rects contains all the faces detected

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        (lStart, lEnd) = (42, 48) #눈좌표
        (rStart, rEnd) = (36, 42)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftbrow = shape[24]
        rightbrow = shape[19]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)


        (lx, ly) = leftbrow
        cv2.circle(frame, (lx, ly), 1, (0, 255, 0), 3)

        (rx, ry) = rightbrow
        cv2.circle(frame, (rx, ry), 1, (0, 0, 255), 3)

        if horizon == False or retrial_brow == True:
            retrial_brow == False
            brow_basis = ly-ry
            eyebrowError['state'] = NORMAL

        if horizon == True:
            if leftEAR < 0.18 or rightEAR < 0.18:

                if abs(ly - ry + brow_basis) > 6:
                    if ly - ry > 0:
                        leftState = "close"
                        rightState = "None"

                    else:
                        rightState = "close"
                        leftState = "None"
                else:
                    leftState = "close"
                    rightState = "close"
            else:
                rightState = "None"
                leftState = "None"

def faceDetect_cascade(gray, frame):
    global eye_detect, horizon, leftState, rightState

    # 얼굴 탐지
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y, w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)  #사각형 범위

        if w < 250:
            cv2.putText(frame, "closer please", (x-5, y-5), font, 0.5, (255,255,0),2)
            eye_detect = False
            continue
        else:
            cv2.putText(frame, "Good", (x-5, y-5), font, 0.5, (255,255,0),2)
            eye_detect = True


        if eye_detect:  #눈찾기
            roi_gray = gray[y:int(y+h/2), x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            width_centor = x+w/2
            current = None #True - right / False - left

            eyes_num = 0
            for (ex, ey, ew, eh) in eyes:
                if eyes_num > 1:
                    continue
                eyes_num = eyes_num + 1
                pupil_frame = roi_gray[ey: ey+eh, ex: ex+ew]

                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey +eh), (255, 0, 0), 2)

                if width_centor > x + ex + ew/2: #show right eyes
                    right_gray_roi, contours = pupil_detect(pupil_frame, threshold_value)
                    current = True

                else: #how left eyes
                    left_gray_roi, contours = pupil_detect(pupil_frame, threshold_value)
                    current = False

                if len(contours) != 0:
                    (cx, cy, cw, ch) = cv2.boundingRect(contours[0]) #제일 큰 conture를 출력하자
                    centerX = int((cx*2 + cw)/2)
                    centerY = int((cy*2 + ch)/2)

                    cv2.circle(roi_color, (ex + centerX, ey + centerY), 1, (0, 0, 255), 3)
                    cv2.rectangle(roi_color, (ex + cx, ey + cy), (ex + cx+cw, ey + cy+ch), (255, 0, 0), 2)

                else:
                    if current == True:
                        rightState = "close"
                    else:
                        leftState = "close"

                if current == True:
                    if rightState == "close":
                        continue
                    if abs(ew/2 - centerX) > 5:
                        if ew/2 - centerX > 0: #right
                            rightState = "right"
                        else:
                            rightState = "left"
                    else:
                        rightState = "None"
                else:
                    if leftState == "close":
                        continue
                    if abs(ew/2 - centerX) > 5:
                        if ew/2 - centerX > 0: #right
                            leftState = "right"
                        else:
                            leftState = "left"
                    else:
                        leftState = "None"

def pupil_detect(gray_roi, threshold_value):
    rows, cols = gray_roi.shape
    gray_roi = cv2.GaussianBlur(gray_roi, (7,7), 0) #잡음제거

    #눈동자 검출
    _, threshold = cv2.threshold(gray_roi, threshold_value, 255, cv2.THRESH_BINARY_INV)

    #눈동자 좌표 가져와서 표시
    contours, none = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #눈동자만 표시
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    return threshold, contours

def eye_aspect_ratio(eye):
    A = dist(eye[1], eye[5])
    B = dist(eye[2], eye[4])
    C = dist(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


#Button event handling
def pupil_thresholdUP():
    global threshold_value
    threshold_value = threshold_value + 5
    th_label.config(text=str(threshold_value))

def pupil_thresholdDown():
    global threshold_value
    threshold_value = threshold_value - 5
    th_label.config(text=str(threshold_value))

def setBrow():
    global horizon, brow_basis
    horizon = True
    retrial_brow = True
    brow_label.config(text=str(brow_basis))
    settingEnd['state'] = NORMAL


# Drone Stream
def startDrone():
    global showStream
    
    print("startDrone")
    if showStream == 0:
        showStream = Thread(target = droneSteram)
        print(showStream)
        showStream.start()

def droneSteram():
    global tello, tello_frame_read
    print("droneStream")

    tello = Tello()
    tello.connect()
    tello.streamon()
    tello_frame_read = tello.get_frame_read()
    time.sleep(5)
    print("get ready")

    while True:
        telloImg = tello_frame_read.frame
        telloImg2 = cv2.resize(telloImg, (w, h)) 
        cv2.imshow("Drone View", telloImg2)
        cv2.waitKey(1)

def checkDroneState():
    global flying, tello_frame_read

    if tello_frame_read != None:
        take_picture['state'] = NORMAL


    if flying == True:
        up_btn['state'] = NORMAL
        down_btn['state'] = NORMAL
    else:
        up_btn['state'] = DISABLED
        down_btn['state'] = DISABLED

def droneUp():
    global flying, tello

    if flying == True:
        tello.move_up(20)
        print("move_up")
        Current_motion.config(text="move_up")

def droneDown():
    global flying, tello

    if flying == True:
        tello.move_down(20)
        print("move_down")
        Current_motion.config(text="move_down")

def nothing():
    pass


#function to order points to proper rectangle
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


#function to transform image to four points
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)

    # # multiply the rectangle by the original ratio
    # rect *= ratio

    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


#function to find two largest countours which ones are may be
#  full image and our rectangle edged object
def findLargestCountours(cntList, cntWidths):
    newCntList = []
    newCntWidths = []

    #finding 1st largest rectangle
    first_largest_cnt_pos = cntWidths.index(max(cntWidths))

    # adding it in new
    newCntList.append(cntList[first_largest_cnt_pos])
    newCntWidths.append(cntWidths[first_largest_cnt_pos])

    #removing it from old
    cntList.pop(first_largest_cnt_pos)
    cntWidths.pop(first_largest_cnt_pos)

    #finding second largest rectangle
    seccond_largest_cnt_pos = cntWidths.index(max(cntWidths))

    # adding it in new
    newCntList.append(cntList[seccond_largest_cnt_pos])
    newCntWidths.append(cntWidths[seccond_largest_cnt_pos])

    #removing it from old
    cntList.pop(seccond_largest_cnt_pos)
    cntWidths.pop(seccond_largest_cnt_pos)

    print('Old Screen Dimentions filtered', cntWidths)
    print('Screen Dimentions filtered', newCntWidths)
    return newCntList, newCntWidths


def takePicture():
    global tello_frame_read

    img = tello_frame_read.frame

    print("take a picture")
    current_time = time.strftime('%H%M%S')
    cv2.imshow(current_time + ".png", img)

    pic_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pic_gray = cv2.GaussianBlur(pic_gray, (5, 5), 0)

    cv2.namedWindow("Canny Edge")
    cv2.createTrackbar('low threshold', 'Canny Edge', 0, 1000, nothing)
    cv2.createTrackbar('high threshold', 'Canny Edge', 0, 1000, nothing)

    cv2.setTrackbarPos('low threshold', 'Canny Edge', 50)
    cv2.setTrackbarPos('high threshold', 'Canny Edge', 150)

    while True:
        low = cv2.getTrackbarPos('low threshold', 'Canny Edge')
        high = cv2.getTrackbarPos('high threshold', 'Canny Edge')

        img_canny = cv2.Canny(pic_gray, low, high)
        cv2.imshow("Canny Edge", img_canny)

        keypress = cv2.waitKey(1)
        if keypress & 0xFF == ord('q'):
            break

    #get contours
    cnts, hierarcy = cv2.findContours(img_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    screenCntList = []
    scrWidths = []
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)  # cnts[1] always rectangle O.o
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        screenCnt = approx
        # print(len(approx))

        if (len(screenCnt) == 4):

            (X, Y, W, H) = cv2.boundingRect(cnt)
            # print('X Y W H', (X, Y, W, H))
            screenCntList.append(screenCnt)
            scrWidths.append(W)

    screenCntList, scrWidths = findLargestCountours(screenCntList, scrWidths)

    pts = screenCntList[0].reshape(4, 2)
    warped = four_point_transform(img, pts)

    cv2.imshow("warp", warped)
    cv2.waitKey(0)



window = Tk()
window.title("Drone Eyes")
window.geometry("720x650")
window.resizable(False, False)

title = Label(window, text="Setting")
title.pack()

app = Frame(window, bg="white")
app.pack()
lmain = Label(app)
lmain.pack()

th_label = Label(window, text="30")
pupilPlusBtn = Button(window, text = 'pupil ++', command = pupil_thresholdUP)
pupilMinusBtn = Button(window, text = 'pupil --', command = pupil_thresholdDown)

th_label.place(x=110, y=520)
pupilPlusBtn.place(x=50, y=520)
pupilMinusBtn.place(x=135, y=520)

brow_label = Label(window, text="0")
eyebrowError = Button(window, text = 'brow zero adjustment', command = setBrow, state=DISABLED)

settingEnd = Button(window, text = 'START', state=DISABLED, command = startDrone) 

brow_label.place(x=400, y=520)
eyebrowError.place(x=420, y=520)
settingEnd.place(x=630, y=520)

Current_motion = Label(window, text="No Motion")
Current_motion.place(x=300, y=550)

up_btn = Button(window, text = 'UP', state=DISABLED, command =droneUp)
down_btn = Button(window, text = 'DOWN', state=DISABLED, command =droneDown)

up_btn.place(x=400, y=600)
down_btn.place(x=430, y=600)

take_picture = Button(window, text = 'Take a Picture', state=DISABLED, command =takePicture)

take_picture.place(x=600, y=600)


# Capture from camera
cap = cv2.VideoCapture(0)

# function for video streaming
def video_stream():
    global i, time_value
    
    checkDroneState()  
    
    if time_value == int(i):
        time_value = int(i) + 1
        if horizon == True and eye_detect == True:
            selectMotion()
    i += 0.5

    checkChangEyes()

    _, img = cap.read()
    frame = img
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    face_landmark(gray, frame)
    faceDetect_cascade(gray, frame)

    cv2.putText(frame, str(leftState), (300,300), font, 0.5, (255,255,0), 2)
    cv2.putText(frame, str(rightState), (350,300), font, 0.5, (255,255,0), 2)

    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, video_stream) 


video_stream()
window.mainloop()