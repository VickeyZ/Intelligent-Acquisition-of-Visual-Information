import numpy as np
import cv2 
from matplotlib import pyplot as plt
from scipy import optimize

MIN_MATCH_COUNT = 10

img1 = cv2.imread('left08.jpg',0)
image1 = cv2.imread('left08.jpg',1)
img2 = cv2.imread('right08.jpg',0)
image2 = cv2.imread('right08.jpg',1)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0#kd树
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# cal mean color
def mean_color(img,h,w,x,y,size):
    color = []
    for i in range(0,img.shape[0]):
        for j in range(0, img.shape[1]):
            if abs(i-y) < size[0][0] and abs(j-x) < size[0][1] and j < w and j > 0 and i < h and i > 0 :
                color.append(img[i][j])
    # print(color)
    return np.mean(color,axis = 0)

# store all the good matches as per Lowe's ratio test.

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

src_color = []
dst_color = []
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #src_size = np.float32([ kp1[m.queryIdx].size for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #dst_size = np.float32([ kp2[m.trainIdx].size for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    h2, w2 = img2.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    print(h2,w2)
    for i,p in enumerate(src_pts):
        sx = int(p[0][0])
        sy = int(p[0][1])
        dx = int(dst_pts[i][0][0])
        dy = int(dst_pts[i][0][1])
        if (sx < w and sx > 0 and sy < h and sy > 0 and dx < w2 and dx > 0 and dy < h2 and dy > 0):
            # print(image1[sy][sx],image2[dy][dx])
            # print(src_size[i])
            src_color.append(image1[sy][sx])
            dst_color.append(image2[dy][dx])
            # src_color.append(mean_color(image1,h,w,sx,sy,src_size[i]))
            # dst_color.append(mean_color(image2,h,w,dx,dy,dst_size[i]))
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

with open('src.txt', 'w') as file:
    for i in src_color:
        s = str(i[0])+' '+str(i[1])+' '+str(i[2])+'\n'
        file.write(s)

with open('dst.txt', 'w') as file:
    for i in dst_color:
        s = str(i[0])+' '+str(i[1])+' '+str(i[2])+'\n'
        file.write(s)
# plt.imshow(img3, 'gray'),plt.show()
# print(src_size)

############## 优化3*3矩阵 ###############

y = np.array(dst_color)
x = np.array(src_color)

def fun(m,y_,x_):
    W = m.reshape(3,3)
    y_pred = np.dot(x_, W)
    return 0.5*np.sum((y_- y_pred)**2,axis=1)

matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(3,3)

# W = np.dot(x,matrix)
# print(np.sum((y - np.dot(x,matrix))**2,axis=1))
W = optimize.leastsq(fun, matrix,args=(y, x))
print(W)
# [ 0.66175069, -0.11251032, -0.22604791,  0.11485519,  0.86162838,    
#        0.50643543,  0.11604199,  0.13286653,  0.65228846]
h,w = img1.shape
W = [0.98,0.002007,-0.02,-0.014843,0.98,-0.02,-0.019999,-0.01737,0.98]
image_ = np.dot(image1, np.array(W).reshape(1,1,3,3)).reshape(h,w,3)
cv2.imwrite('c_left08.jpg',image_)
# image_ = np.dot(image2, np.array(W[0]).reshape(1,1,3,3)).reshape(h,w,3)
print(image_.shape)

b,g,r = cv2.split(image2)
image2 = cv2.merge([r,g,b])
b,g,r = cv2.split(image_)
image_ = cv2.merge([r,g,b])

plt.subplot(1,2,1)
plt.imshow(image2)
plt.subplot(1,2,2)
plt.imshow(image_/255)
plt.show()
# plt.imshow(image_, 'convert'),plt.show()