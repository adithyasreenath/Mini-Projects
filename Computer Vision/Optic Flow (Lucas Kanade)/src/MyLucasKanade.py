import math
from scipy import signal
from PIL import Image
import numpy as np
from numpy import *
from matplotlib import pyplot as plt
from pylab import *
import cv2
import random

def LK_OpticalFlow(Image1,Image2,):
   	I1 = np.array(Image1)
	I2 = np.array(Image2)
	S = np.shape(I1)
	I1_smooth = cv2.GaussianBlur(I1, (3,3), 0)
	I2_smooth = cv2.GaussianBlur(I2, (3,3), 0)
	Ix = signal.convolve2d(I1_smooth,[[-0.25,0.25],[-0.25,0.25]],'same') + signal.convolve2d(I2_smooth,[[-0.25,0.25],[-0.25,0.25]],'same')
	Iy = signal.convolve2d(I1_smooth,[[-0.25,-0.25],[0.25,0.25]],'same') + signal.convolve2d(I2_smooth,[[-0.25,-0.25],[0.25,0.25]],'same')
	It = signal.convolve2d(I1_smooth,[[0.25,0.25],[0.25,0.25]],'same') + signal.convolve2d(I2_smooth,[[-0.25,-0.25],[-0.25,-0.25]],'same')
	features = cv2.goodFeaturesToTrack(I1_smooth,10000,0.01,10)	
	feature = np.int0(features)
	for i in feature:
		x,y = i.ravel()
		cv2.circle(I1_smooth,(x,y),3,0,-1)
	u = v = np.nan*np.ones(S)
	for l in feature:
		j,i = l.ravel()
		IX = ([Ix[i-1,j-1],Ix[i,j-1],Ix[i-1,j-1],Ix[i-1,j],Ix[i,j],Ix[i+1,j],Ix[i-1,j+1],Ix[i,j+1],Ix[i+1,j-1]])
		IY = ([Iy[i-1,j-1],Iy[i,j-1],Iy[i-1,j-1],Iy[i-1,j],Iy[i,j],Iy[i+1,j],Iy[i-1,j+1],Iy[i,j+1],Iy[i+1,j-1]])
		IT = ([It[i-1,j-1],It[i,j-1],It[i-1,j-1],It[i-1,j],It[i,j],It[i+1,j],It[i-1,j+1],It[i,j+1],It[i+1,j-1]])
		LK = (IX, IY)
		LK = np.matrix(LK)
		LK_T = np.array(np.matrix(LK))
		LK = np.array(np.matrix.transpose(LK)) 
		A1 = np.dot(LK_T,LK)
		A2 = np.linalg.pinv(A1)
		A3 = np.dot(A2,LK_T)
		(u[i,j],v[i,j]) = np.dot(A3,IT)
	plt.subplot(1,1,1)
	plt.title('Optical Flow')
	plt.imshow(I1,cmap = cm.gray)
	for i in range(S[0]):
		for j in range(S[1]):
			if abs(u[i,j])>t or abs(v[i,j])>t:
				plt.arrow(j,i,v[i,j],u[i,j],head_width = 5, head_length = 5, color = "b")
	plt.show()
t = 0.3
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
img1 = cv2.imread('image1.png')
img2 = cv2.imread('image2.png')
Image1 = Image.open('image1.png').convert('L')
Image2 = Image.open('image2.png').convert('L')
cv2.imshow('image1',img1)
cv2.imshow('image2',img2)
LK_OpticalFlow(Image1, Image2)
