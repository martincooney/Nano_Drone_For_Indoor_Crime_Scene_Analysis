#!/usr/bin/env python
import numpy as np
import cv2
import math

print('-----------------------------------------------')
print('-   DETECT DIRECTION BLOODY OBJECT WAS MOVED  -')
print('-        FEB 2025, HH, Martin Cooney          -')
print('-----------------------------------------------')

#Note: while writing this code, inspiration was taken from the following sources for finding contour centers and angles from dot product:
#https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/
#https://stackoverflow.com/questions/66839398/python-opencv-get-angle-direction-of-fitline

#This helper function was also taken and adapted from: https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
def line_intersection(line1, line2):
	xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
	ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

	def det(a, b):
		return a[0] * b[1] - a[1] * b[0]

	div = det(xdiff, ydiff)
	if div == 0:
		return -1, -1  #lines do not intersect

	d = (det(*line1), det(*line2))
	x = det(d, xdiff) / div
	y = det(d, ydiff) / div
	return int(x), int(y)


#This is our core function that processes one image sample
def detect_direction_for_one_sample(inputFileLabel, groundTruthAngle):

	#read in the image
	inputFileName= "input/raw_photos/" + inputFileLabel + "_" + str(groundTruthAngle) + ".png"
	img = cv2.imread(inputFileName)
	print("") 
	print("Processing: inputFileName", inputFileName)
	height, width = img.shape[:2] #image_raw

	#find red (blood)
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	#NOTE: change the values below for color picking depending on illumination, etc.
	mask = cv2.inRange(img_hsv, (150,20,0), (180,255,255))
	cropped = cv2.bitwise_and(img, img, mask=mask)

	#calculate the center of the mask
	ys, xs = np.where(mask > 0)  
	centerOfBlood = (-1,-1)
	if len(xs) > 0 and len(ys) > 0:
		center_x = int(np.mean(xs))
		center_y = int(np.mean(ys))
		centerOfBlood = (center_x, center_y)
		print("Center of white pixels:", centerOfBlood)
		cv2.circle(img, centerOfBlood, 7, (0, 255, 0), -1)
	else:
		print("No blood pixels found.")

	#calculate and process contours
	listOfCentroids=[]
	lastGoodContour = None
	(cnts, *_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	for c in cnts:
		area = cv2.contourArea(c)
		if ( area > 200.0): #check that we have a reasonably sized contour

			lastGoodContour=c
			M = cv2.moments(c)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			listOfCentroids.append([cX,cY])

			# draw the contour and center of the shape on the image
			cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
			cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)

	print("Number of reasonably sized contours", len(listOfCentroids))

	if(not listOfCentroids):
		print("No contours detected! Let's return a default guess, a horizontal line along x axis from center with angle 0")
		x0 = width/2
		y0 = height/2
		vx = [x0+1]
		vy = [y0]

	elif(len(listOfCentroids)==1):
		print("Only 1 contour! Special treatment is needed to fit a line")

		M = cv2.moments(lastGoodContour)
		x0 = [int(M["m10"] / M["m00"])]
		y0 = [int(M["m01"] / M["m00"])]
		theta = 0.5*np.arctan2(2*M["mu11"],M["mu20"]-M["mu02"])
		vx = [np.cos(theta)]
		vy = [np.sin(theta)]
	
	else: #multiple contours were found, fit a line through the centroids
		centroids_contour = np.array(listOfCentroids).reshape((-1,1,2)).astype(np.int32)
		[vx,vy,x0,y0] = cv2.fitLine(centroids_contour, cv2.DIST_L2,0,0.01,0.01)
		print("Line fit:", vx[0], vy[0], x0[0], y0[0])

	#mark the center of the line of best fit
	cv2.circle(img, (int(x0[0]), int(y0[0])), 10, (255, 0, 0), -1)

	endpoint1= [x0[0], y0[0]]
	endpoint2= [x0[0]+vx[0], y0[0]+vy[0]]
	endpoint_windowTop1 = [0, 0]
	endpoint_windowTop2 = [1, 0]
	endpoint_windowBottom1 = [0, height-1] 
	endpoint_windowBottom2 = [1, height-1]
	endpoint_windowLeft1 = [0, 0]
	endpoint_windowLeft2 = [0, 1]
	endpoint_windowRight1 = [width-1, 0] 
	endpoint_windowRight2 = [width-1, 1]

	#find intersection with the window top and bottom, and window left and right
	intersectionWithTop= line_intersection( (endpoint1,endpoint2), (endpoint_windowTop1, endpoint_windowTop2) )
	intersectionWithBottom= line_intersection( (endpoint1,endpoint2), (endpoint_windowBottom1, endpoint_windowBottom2) )
	intersectionWithLeft= line_intersection( (endpoint1,endpoint2), (endpoint_windowLeft1, endpoint_windowLeft2) )
	intersectionWithRight= line_intersection( (endpoint1,endpoint2), (endpoint_windowRight1, endpoint_windowRight2) )

	#next, find which of the four points are good
	validPoints=[]
	intersectionWithinImage_top=True
	intersectionWithinImage_bottom=True
	intersectionWithinImage_left=True
	intersectionWithinImage_right=True

	if(intersectionWithTop[0] < 0 or intersectionWithTop[1] < 0 or intersectionWithTop[0] > width or intersectionWithTop[1] > height):
		intersectionWithinImage_top = False
	else:
		validPoints.append(intersectionWithTop)
	if(intersectionWithBottom[0] < 0 or intersectionWithBottom[1] < 0 or intersectionWithBottom[0] > width or intersectionWithBottom[1] > height):
		intersectionWithinImage_bottom = False
	else:
		validPoints.append(intersectionWithBottom)
	if(intersectionWithLeft[0] < 0 or intersectionWithLeft[1] < 0 or intersectionWithLeft[0] > width or intersectionWithLeft[1] > height):
		intersectionWithinImage_left = False
	else:
		validPoints.append(intersectionWithLeft)
	if(intersectionWithRight[0] < 0 or intersectionWithRight[1] < 0 or intersectionWithRight[0] > width or intersectionWithRight[1] > height):
		intersectionWithinImage_right = False
	else:
		validPoints.append(intersectionWithRight)

	#determine which is the start point of the line and which is the end point
	if (len(validPoints)==2 and not centerOfBlood[0] == -1):
		d1= math.dist(validPoints[0], centerOfBlood)
		d2= math.dist(validPoints[1], centerOfBlood)
		if(d1 < d2): #start at the top
			arrowStart = validPoints[0]
			arrowEnd = validPoints[1]
		else:
			arrowStart = validPoints[1]
			arrowEnd = validPoints[0]
		cv2.arrowedLine(img, arrowStart, arrowEnd, (0,0,0))
	else:
		print("Complex case or error? Let's ignore this for now")

	x_axis = np.array([1, 0]) 
	direction_line = np.array([vx[0], vy[0]]) 

	dot_product = np.dot(x_axis, direction_line) 
	angle_to_x_axis = math.degrees(np.arccos(dot_product))
	print("angle_to_x_axis", angle_to_x_axis)

	#next, check the quadrant
	#this is a bit complex because in opencv y=0 is at the top and the fit line unit vector can be headed either way

	if(arrowStart[0] <= arrowEnd[0] and arrowStart[1] >= arrowEnd[1]): #quadrant 1 opencv
		print("Quadrant 1")
	elif(arrowStart[0] >= arrowEnd[0] and arrowStart[1] >= arrowEnd[1]): #quadrant 2
		print("Quadrant 2")
		angle_to_x_axis = 180- angle_to_x_axis
	elif(arrowStart[0] >= arrowEnd[0] and arrowStart[1] <= arrowEnd[1]): #quadrant 3
		print("Quadrant 3")
		angle_to_x_axis += 180
	else:								     #quadrant 4
		print("Quadrant 4")
		angle_to_x_axis = 360-angle_to_x_axis
	print("Angle adjusted by quadrant:", angle_to_x_axis)

	#now check how correct we were
	rough_angle = -1
	min_index = -1
	min_distance = 360
	best_label = ""
	main_directions = [0, 90, 180, 270, 360]
	main_labels = ["right", "up", "left", "down", "right"]

	for idx, x in enumerate(main_directions):
		d =  abs(x - angle_to_x_axis)
		if (d < min_distance):
			min_distance = d
			rough_angle = x
			best_label = main_labels[idx]
	if(rough_angle==360):
		rough_angle = 0

	weWereCorrect = (rough_angle == groundTruthAngle)
	errorToGroundTruth = abs(angle_to_x_axis-groundTruthAngle)

	print("Result")
	print("We were correct?", weWereCorrect)
	print("Error:", errorToGroundTruth)
	print("Angle_to_x_axis:", angle_to_x_axis)
	print("Predicted rough output:", rough_angle)
	print("Predicted text label:", best_label)

	cv2.imwrite("output/" + inputFileLabel + "_" + str(groundTruthAngle) + "_output.png", img)

	output = [weWereCorrect, errorToGroundTruth, angle_to_x_axis, rough_angle, best_label]
	return output


#get results for all data samples
myResults = []
for i in range(5):
	dataSampleLabel = "d" + str(i+1)
	myResults.append(detect_direction_for_one_sample(dataSampleLabel, 0))
	myResults.append(detect_direction_for_one_sample(dataSampleLabel, 90))
	myResults.append(detect_direction_for_one_sample(dataSampleLabel, 180))
	myResults.append(detect_direction_for_one_sample(dataSampleLabel, 270))

totalCorrect=0
totalError=0.0
totalErrorCorrectOnly=0.0

print("")
print("output = [weWereCorrect, errorToGroundTruth, predicted angle_to_x_axis, predicted rough_angle, predicted best_label]")

for i in range(len(myResults)):
	if(myResults[i][0]):
		totalCorrect += 1
		totalErrorCorrectOnly += myResults[i][1]
	else:
		if(i%4==0):
			angleLabel = 0
		elif(i%4==1):
			angleLabel = 90
		elif(i%4==2):
			angleLabel = 180
		elif(i%4==3):
			angleLabel = 270
		print("Wrong case:", i, "d", int(i/4)+1, angleLabel, myResults[i])
	totalError+=myResults[i][1]

accuracy= float(totalCorrect)/(float(len(myResults)))
averageError= totalError/(float(len(myResults)))
averageErrorCorrectOnly=totalErrorCorrectOnly/float(totalCorrect)

#output the results
print("")
print("Accuracy:", accuracy)
print("Average Error in Angle:", averageError)
print("Average Error (Correct Only):", averageErrorCorrectOnly)
print("Number of results:", len(myResults))

resultsBySampleType=[]
for i in range(5):
	sum=0.0
	for j in range(4):
		if(myResults[(i*4)+j][0]):
			sum+=1.0
	ave= sum/4.0
	resultsBySampleType.append(ave)
print("Results By Sample Type:", resultsBySampleType)

#To go further
#one could make a confusion matrix, or check misclassified samples, etc.
#we could also try using more angles, like:
#main_directions=[0, 45, 90, 135, 180, 225, 270, 315, 360]
#main_labels=["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right", "right"]


