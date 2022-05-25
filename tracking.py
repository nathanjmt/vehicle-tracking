import cv2
import numpy as np
import math

# get user to input video and check if video is successfully opened
openedVid = False
while not openedVid:
    try:
        video_path = input("Enter video path: ")
        cap = cv2.VideoCapture(video_path)
        openedVid = True
    except:
        print("Error opening video file, check if path entered is correct")

#D:\Personal Projects\improved vehicle tracking\Videos\Videos\Traffic_Laramie_1.mp4
#C:\Users\natha\Desktop\lecture notes\Object-tracking-from-scratch-source_code\source_code\los_angeles.mp4
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print("Enter an integer for x-axis boundaries, from 0 to " + str(int(width)))
x_valid = False
while not x_valid: 
    try:
        x_start = int(input("From: "))
        x_end = int(input("To: "))
        x_valid = True
    except ValueError:
        print("Please only input integers")


print("Enter an integer for y-axis boundaries, from 0 to " + str(int(height)))
y_valid = False
while not y_valid: 
    try:
        y_start = int(input("From: "))
        y_end = int(input("To: "))
        y_valid = True
    except ValueError:
        print("Please only input integers")

size_valid = False
while not size_valid: 
    try:
        size = int(input("Enter rough vehicle pixel size: "))
        size_valid = True
    except ValueError:
        print("Please only input integers")

  
initial_frame = None


#start a loop to keep reading frames from the video
while True:
    ret, frame = cap.read()
    
    #check if the video still has frames
    if ret == True:

        #convert the frame to grayscale and reduce noise by using Gaussian Blur
        gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur_frame=cv2.GaussianBlur(gray_frame,(15,15), 0)
        
        # The first frame captured is the baseline image
        if initial_frame is None:
            initial_frame = blur_frame
            continue

        # use absdiff for frame differencing between the baseline frame and the new frame
        #the frames are being spliced to ensure that motion detection is only performed on the main road, as specified
        delta_frame=cv2.absdiff(initial_frame[y_start:y_end, x_start:x_end],blur_frame[y_start:y_end, x_start:x_end])
        # delta frameis converted into a binary image
        
        # If a particular pixel value is greater than a certain threshold (specified by us here as 10),

        threshold_frame=cv2.threshold(delta_frame,10,255, cv2.THRESH_BINARY)[1]
        #if a pixel value is greater than the threshold we set (10) it will be white. else it will be black


        # The cv2.findContours() method identifies all the contours in the image.
        (contours,ret)=cv2.findContours(threshold_frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)      
        
        for c in contours:
            # contourArea() method filters out any small contours
            # this is to ensure that only cars are being detected
            if cv2.contourArea(c) < size:
                continue
            #draw rectangles around the objects we have detected
            #due to the frame splicing earlier, we have to shift the rectangles by the same values
            (x, y, w, h)=cv2.boundingRect(c)
            cv2.rectangle(frame[y_start:y_end, x_start:x_end], (x, y), (x+w, y+h), (0,255,0), 1)
            

        cv2.imshow("Frame", frame)
        cv2.imshow("threshold", threshold_frame)
        cv2.imshow("delta", delta_frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
        
    #end the video when there are no more frames  
    else:
        break

# After the loop release the video object    
cap.release()

# Destroy all the windows
cv2.destroyAllWindows