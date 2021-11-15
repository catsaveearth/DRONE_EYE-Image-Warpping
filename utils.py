#201935023 김수현
from djitellopy import Tello
import cv2
import time


def initTello():
    myDrone = Tello()
    myDrone.connect()

    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.speed = 0

    print("\n * Drone battery percentage : " + str(myDrone.get_battery()) + "%")
    myDrone.streamoff()
    
    return myDrone


def moveTello(myDrone):

    myDrone.takeoff()
    time.sleep(5)

    myDrone.move_back(50)
    time.sleep(5)
    myDrone.rotate_clockwise(360)
    time.sleep(5)
    myDrone.move_forward(50)
    time.sleep(5)

    myDrone.flip_right()
    time.sleep(5)
    myDrone.flip_left()
    time.sleep(5)

    myDrone.land()
    time.sleep(5)




def telloGetFrame(myDrone, w = 360, h = 240):
    myFrame = myDrone.get_frame_read() 
    myFrame = myFrame.frame #numpy.ndarray

    img = cv2.resize(myFrame, (w, h)) #numpy.ndarray

    return img
#201935023 김수현