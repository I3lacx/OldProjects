import numpy as np
import cv2

num_examples = 1000
pic_width = 28
pic_height = 28


#returnes array of pictures and positions of given batch size
def next_batch(batch_size):
    pic_arr = []
    pos_arr = []

    for _ in range(batch_size):
        pic, pos = create_sample(pic_width, pic_height)
        pic_arr.append(pic)
        pos_arr.append(pos)

    return pic_arr, pos_arr

#returns coorinates (x1,y1,x2,y2) of rectangle in range of pic with border
#with minimal w and h
def random_rect(pic_w, pic_h, border, min_w, min_h):
    r_width_range = pic_w - border
    r_height_range = pic_h - border

    r_x = np.random.random_integers(border, r_width_range - (min_w + border))
    r_y = np.random.random_integers(border, r_height_range - (min_h + border))

    r_x2 = np.random.random_integers(r_x + min_w, r_width_range)
    r_y2 = np.random.random_integers(r_y + min_h, r_height_range)

    return r_x, r_y, r_x2, r_y2

def randomColor():
    r = np.random.random_integers(0,255)
    g = np.random.random_integers(0,255)
    b = np.random.random_integers(0,255)

    return int(b),int(g),int(r)

def create_sample(pic_w, pic_h):
    #pre defined rectangle size
    x,y,x1,y1 = random_rect(pic_w,pic_h,1,2,2)
    pic = white_pic(pic_w,pic_h) # without color first
    #creates rectangle on picture
    cv2.rectangle(pic, (x,y), (x1,y1), 0, thickness = -1)
    rec_pos = [x,y,x1,y1]

    return pic, rec_pos

#returns a cv2 image completly white
def white_pic(w,h,color=False):
    if(color):
        pic = np.zeros((h,w,3), np.uint8)
        for i in range(len(pic)):
            for j in range(len(pic[0])):
                for c in range(3):
                    pic[i][j][c] = 255
        return pic

    pic = np.zeros((h,w))

    #set everyting white
    for i in range(len(pic)):
        for j in range(len(pic[0])):
            pic[i][j] = 255

    return pic


#img settings
width = 28
height = 28

for i in range(100):

    img, pos = create_sample(width,height)
    cv2.imshow('rect', img)

    if cv2.waitKey(25) == ord('q'):
        cv2.destroyAllWindows()
        break
