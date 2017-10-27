import numpy as np
from PIL import ImageGrab
import cv2
import win32api, win32con

def get_cords():
    x,y = win32api.GetCursorPos()
    return x,y


while(True):
    mouse_X,mouse_Y = get_cords()
    printscreen_pil = ImageGrab.grab(bbox=(mouse_X-100,mouse_Y-100,mouse_X+100,mouse_Y+100))
    cv2.imshow('window', np.array(printscreen_pil))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
