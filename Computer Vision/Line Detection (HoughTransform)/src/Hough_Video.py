import cv2
import numpy as np
global dst,img,x,y,z

def CannyDet(x):
	dst = cv2.Canny(img, 0, x, 3)
	cv2.imshow('AS3', dst)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
def HoughLines(img,angle,threshold):
	lines = []
	y_idxs, x_idxs = np.nonzero(img)
	thetas = angle*(np.arange(-(np.pi/(angle*2)), (np.pi/(angle*2))))
	cos_t = np.cos(thetas)
	sin_t = np.sin(thetas)
	len_t = len(thetas)
	all_lines = {}
	for t in range(len_t):
		for i in range(y_idxs.size):
			r = round(x_idxs[i]*cos_t[t] + y_idxs[i]*sin_t[t])
			if (r,thetas[t]) in all_lines.keys():
				all_lines[(r,thetas[t])] = all_lines[(r,thetas[t])] + 1
			else:
				all_lines[(r,thetas[t])] = 1	
	all_lines = {k:v for (k,v) in all_lines.items() if v >= threshold}
	return np.asarray(list(all_lines.copy().keys()))

def HoughTransformbin(y):
	lines = cv2.HoughLines(dst, 1, (np.pi)/180, 100, y, 0 )[:,0,:]
	drawlines(lines)
	
def HoughTransformPeak(z):
	z = int(z)
	lines = cv2.HoughLines(dst, 1, (np.pi)/180, z, 30, 0 )[:,0,:]
#	lines = HoughLines(dst, (np.pi)/180, z)
	drawlines(lines)

	#my implementation
#	lines = HoughLines(dst,(np.pi)/180,100)
#	minLineLength = 100
#	maxLineGap = 10
#	lines = cv2.HoughLinesP(dst , 1, np.pi/180, 100, minLineLength, maxLineGap)
#	for x1,y1,x2,y2 in lines[0]:
#		cv2.line(src,(x1,y1),(x2,cdy2),(0,255,0),2)
def drawlines(lines):
#	img1 = cv2.imread('image1.jpg',0)
	img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	for line in lines:
	 rho = line[0]
	 theta = line[1]
	 a = np.cos(theta)
	 b = np.sin(theta)
	 
	 x0 = a*rho
	 y0 = b*rho
	 
	 x1 = int(np.round(x0 + 1000*(-b)))
	 y1 = int(np.round(y0 + 1000*(a)))
	 x2 = int(np.round(x0 - 1000*(-b)))
	 y2 = int(np.round(y0 - 1000*(a)))
 	
 	 cv2.line(img1,(x1,y1),(x2,y2),(0,255,255), 1)
	
#	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	img1 = cv2.resize(img1,(800,600))
	cv2.imshow('AS3', img1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#img1 = cv2.imread('image1.jpg',0)
#img = cv2.resize(img1,(800,600))
#dst = cv2.Canny(img, 50, 200, 3)
#cv2.namedWindow('AS3')
#cv2.imshow('AS3', img)
#cv2.createTrackbar('Edge','AS3', 0, 150, CannyDet)
#cv2.createTrackbar('Bin_Size','AS3', 0, 100, HoughTransformbin)
#cv2.createTrackbar('Peak','AS3', 0, 200, HoughTransformPeak)
#
#cv2.waitKey(0)

cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img1,(800,600))
	dst = cv2.Canny(img, 50, 200, 3)
	cv2.namedWindow('AS3')
	cv2.imshow('AS3', img)
	cv2.createTrackbar('Edge','AS3', 0, 150, CannyDet)
	cv2.createTrackbar('Bin_Size','AS3', 0, 100, HoughTransformbin)
	cv2.createTrackbar('Peak','AS3', 0, 200, HoughTransformPeak)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()