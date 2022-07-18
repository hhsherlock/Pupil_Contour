import cv2 as cv
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial import distance

# write data.csv file
def write_data(data):
	f = open('data.csv', 'w')
	writer = csv.writer(f)
	writer.writerow(data)
	f.close()

# use haar cascade crop the eye regions
def haar_eye(gray):
	haar_cascade = cv.CascadeClassifier('haar_eye.xml')
	eyes_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1,
	minNeighbors = 7)
	for (x,y,w,h) in eyes_rect:
		if w > 60 and w < 80:
			eyes_roi_w.append(w)
			eyes_roi.append(gray[y:y+h, x:x+w])
	return eyes_roi, eyes_roi_w

# detect the max and min pixel values and locations
def max_min(pic):
	min_val, max_val, min_indx, max_indx = cv.minMaxLoc(pic)
	dots = pic.copy()
	# # unquote to show the dots in pictures
	# cv.circle(dots, min_indx, 2, (255,0,0), thickness = -1)
	# cv.circle(dots, max_indx, 2, (255,0,0), thickness = -1)
	# cv.imshow('Add Dots', dots)
	return min_val,max_val,min_indx,max_indx



capture = cv.VideoCapture('photos/eye-v-short.mov')

c = 1
eyes_roi = []
eyes_roi_w = []
area = []


while True:
	
	success, frame = capture.read()
	if not success:
		break

	# c%5 is to get one frame every five frames
	if c%5 == 0:
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		haar_eye(gray)

		# iterate every eye region got from haar cascade 
		for x,res in enumerate(eyes_roi):
			# cv.imshow('Original' + str(x),res)

			eroded = cv.erode(res, (3,3), iterations=5)
			# cv.imshow('Eroded' + str(x), eroded)

			dilated = cv.dilate(eroded, (3,3), iterations=5)
			# cv.imshow('Dilated' + str(x), dilated)

			# Increase contrast
			min_val,max_val,min_indx,max_indx = max_min(dilated)
			cal_range = max_val - min_val
			for i in range(0, dilated.shape[0]-1):
				for j in range(0, dilated.shape[1]-1):
					dilated[i,j] = (dilated[i,j] - min_val)/cal_range*255
			# cv.imshow('Contrast' + str(x), dilated)


			blur = cv.GaussianBlur(dilated, (3,3), cv.BORDER_DEFAULT)
			# cv.imshow('Blur' + str(x), blur)


			canny = cv.Canny(dilated, 0, 255)
			# cv.imshow('Canny Edges' + str(x), canny)


			# contour
			contours, hierarchies = cv.findContours(canny, 
			cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

			# # unquote to show the contours after canny
			# blank = np.zeros(canny.shape, dtype = 'uint8')
			# cv.drawContours(blank, contours, -1, (255,0,0),1)
			# cv.imshow('Contour' + str(x), blank)


			# calculate the distance between each contour centroid and min -
			# - pixel value (the darkest point in the pupil)
			calculation = []
			for contour in contours:
				M = cv.moments(contour)
				if M["m00"] != 0:
					center_X = int(M["m10"] / M["m00"])
					center_Y = int(M["m01"] / M["m00"])
					contour_center = (center_X, center_Y)
					distance_to_center = distance.euclidean(contour_center, 
						min_indx)
					calculation.append(distance_to_center)

			# the contour that has the minmum distance to the min pixel value

			contour_index = calculation.index(min(calculation))

			chosen = dilated.copy()
			cv.drawContours(chosen, contours, contour_index, (255,0,0), 1)
			cv.imshow('Chosen Contour', chosen)
			area_data = cv.contourArea(contours[contour_index])
			if area_data > 200 and area_data < 330:
				area.append(area_data)

	c = c+1
	if cv.waitKey(20) & 0xFF==ord('d'):
			break

write_data(area)

# line plot
x = []
for i in range(1, len(area)+1):
	x.append(i)
	i = i + 1

plt.plot(x, area)
plt.xlabel('Frame')
plt.ylabel('Pixel Amount')
plt.show()

cv.waitKey(0)
capture.release()
cv.destroyAllWindows()

