import cv2
import numpy as np
from matplotlib import pyplot as plt
import funda as fd
global pts1
global pts2
global pts3
global pts4
pts3=[]
pts4=[]
pts5=[]
pts6=[]

def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        cv2.line(img1, (x0,y0), (x1,y1), color,1)
        cv2.circle(img1,tuple(pt1),5,color,-1)
        cv2.circle(img2,tuple(pt2),5,color,-1)

    return img1,img2

def my_mouse_callbackL(event,x,y,flags,param):
 if event == cv2.EVENT_LBUTTONDOWN:
     cv2.circle(img1,(x, y), 5, (0, 0, 255), -1)
     pts3.append([x,y])
     pts5.append([x,y])
     print("You clicked on image1 at points : ",x, y)

  
  
  
def my_mouse_callbackR(event,x,y,flags,param):
 if event == cv2.EVENT_LBUTTONDOWN:
     cv2.circle(img2,(x, y), 5, (0, 0, 255), -1)
     pts4.append([x,y])
     pts6.append([x,y])
     print("You clicked on image2 at points : ",x, y)
     
     
     
     
def my_mouse_callbackL1(event,x,y,flags,param):
 if event == cv2.EVENT_LBUTTONDOWN:
     cv2.circle(img1,(x, y), 5, (0, 0, 255), -1)
     index1=pts5.index([x,y])
     img7,img8=draw_single_Line(imgy,imgx,lines2,pts6[index1],pts5[index1],index1)
     while True:
        cv2.imshow("imagey",img7)
        if cv2.waitKey(1) & 0xFF == ord('q'):
         break
  
  
  
def my_mouse_callbackR1(event,x,y,flags,param):
 if event == cv2.EVENT_LBUTTONDOWN:
     cv2.circle(img2,(x, y), 5, (0, 0, 255), -1)
     index2=pts6.index([x,y])
     img9,img10=draw_single_Line(imgx,imgy,lines1,pts5[index2],pts6[index2],index2)
     while True:
        cv2.imshow("imagex",img9)
        if cv2.waitKey(1) & 0xFF == ord('q'):
         break

def draw_single_Line(img1,img2,lines,pts1,pts2,indexval):
    
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    lines = [lines[indexval]]
    for r in lines:
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        cv2.line(img1, (x0,y0), (x1,y1), color,1)
        cv2.circle(img1,tuple(pts1),5,color,-1)
        cv2.circle(img2,tuple(pts2),5,color,-1)
        
    return img1,img2

#Load Images
img1 = cv2.imread('left.jpg',0)  
img2 = cv2.imread('right.jpg',0) 
imgx = cv2.imread('left.jpg',0)  
imgy = cv2.imread('right.jpg',0)
 
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image1',my_mouse_callbackL)
cv2.setMouseCallback('image2',my_mouse_callbackR)
while True:
 cv2.imshow("image1",img1)
 cv2.imshow("image2",img2)
 if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pts3=np.float32(np.array(pts3))
pts4=np.float32(np.array(pts4))
X = fd.fundamental_matrix(pts3,pts4)
#X, mask = cv2.findFundamentalMat(pts3,pts4,cv2.FM_RANSAC)

print("The ESTIMATED FUNDAMENTAL MATRIX is :",X)
# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts4.reshape(-1,1,2), 2,X)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts3,pts4)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts3.reshape(-1,1,2), 1,X)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts4,pts3)

cv2.destroyAllWindows()
#cv2.imshow("image1",img3)
#cv2.imshow("image2",img5)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

cv2.namedWindow('imagex',cv2.WINDOW_NORMAL)
cv2.namedWindow('imagey',cv2.WINDOW_NORMAL)

print("The coordinates of epipoles of 1 are:",pts5)
print("The coordinates of epipoles 2 are:",pts6)

cv2.setMouseCallback('imagex',my_mouse_callbackL1)
cv2.setMouseCallback('imagey',my_mouse_callbackR1)
while True:
 cv2.imshow("imagex",img4)
 cv2.imshow("imagey",img6)
 if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("The coordinates of epipoles of 1 are:",pts5)
print("The coordinates of epipoles 2 are:",pts6)


