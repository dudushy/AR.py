# Imports
# sourcery skip: for-append-to-extend, list-comprehension
import cv2
import numpy as np

# Variables
capture = cv2.VideoCapture(0)
targetImg = cv2.imread('target.jpg')
#targetImg = cv2.imread('target.jpeg')
#targetImg = cv2.imread('target.png')
video = cv2.VideoCapture('video.mp4')

# Get the 1st frame of the video.mp4
success, videoImg = video.read()
# Get target image heigth and width
h, w, c = targetImg.shape
# Resize the 1st frame of the video.mp4
videoImg = cv2.resize(videoImg, (h, w))

orb = cv2.ORB_create(nfeatures=1000)
keyPoint1, descriptor1 = orb.detectAndCompute(targetImg, None)
#targetImg = cv2.drawKeypoints(targetImg, keyPoint1, None)

# Main
while True:
    success, webcamImg = capture.read()
    keyPoint2, descriptor2 = orb.detectAndCompute(webcamImg, None)
    #webcamImg = cv2.drawKeypoints(webcamImg, keyPoint2, None)

    matches = cv2.BFMatcher().knnMatch(descriptor1, descriptor2, 2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))

    # Displays the target image + 1st frame of video.mp4 + webcam
    cv2.imshow('target', targetImg)
    cv2.imshow('video', videoImg)
    cv2.imshow('webcam', webcamImg)
    cv2.waitKey(0)