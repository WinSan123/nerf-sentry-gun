# Python program to implement
# Webcam Motion Detector

#from picamera.array import PiRGBArray
#from picamera import PiCamera
import warnings
#import imutils
import json
import time
import cv2

# filter warnings, load the configuration and initialize the Dropbox
# client
warnings.filterwarnings("ignore")

avg = None

# Capturing video
video = cv2.VideoCapture(0)

# Infinite while loop to treat stack of image as video
while True:
	# Reading frame(image) from video
	check, frame = video.read()


	# Converting color image to gray_scale image
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Converting gray scale image to GaussianBlur
	# so that change can be find easily
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the average frame is None, initialize it
	if avg is None:
		avg = gray.copy().astype("float")
		continue

	# accumulate the weighted average between the current frame and
	# previous frames, then compute the difference between the current
	# frame and running average
	cv2.accumulateWeighted(gray, avg, 0.5)
	diff_frame = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

	# If change in between static background and
	# current frame is greater than 30 it will show white color(255)
	thresh_frame = cv2.threshold(diff_frame, 5, 255, cv2.THRESH_BINARY)[1]
	thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)

	# Finding contour of moving object
	cnts,_ = cv2.findContours(thresh_frame.copy(),
					cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for contour in cnts:
		if cv2.contourArea(contour) < 10000:
			continue

		(x, y, w, h) = cv2.boundingRect(contour)
		# making green rectangle arround the moving object
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

	# Displaying image in gray_scale
	cv2.imshow("Gray Frame", gray)

	# Displaying the difference in currentframe to
	# the staticframe(very first_frame)
	cv2.imshow("Difference Frame", diff_frame)

	# Displaying the black and white image in which if
	# intensity difference greater than 30 it will appear white
	cv2.imshow("Threshold Frame", thresh_frame)

	# Displaying color frame with contour of motion of object
	cv2.imshow("Color Frame", frame)

	key = cv2.waitKey(1)
	# if q entered whole process will stop
	if key == ord('q'):
		break

video.release()

# Destroying all the windows
cv2.destroyAllWindows()
