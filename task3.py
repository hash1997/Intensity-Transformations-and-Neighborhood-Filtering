import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img_orig = cv.imread('highlights_and_shadows.png', cv.IMREAD_GRAYSCALE)
gamma = 2

table = np.array([(i/255.0)**(gamma)*255.0 for i in np.arange(0,256)]).astype('uint8')
img_gamma = cv.LUT(img_orig, table)
img_orig = cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)
img_gamma = cv.cvtColor(img_gamma, cv.COLOR_BGR2RGB)
f, axarr = plt.subplots(3,2)
axarr[0,0].imshow(img_orig)
axarr[0,1].imshow(img_gamma)
plt.show()

color = ('b', 'a', 'L')
for i, c in enumerate(color):
	hist_orig = cv.calcHist([img_orig] ,[i] ,None ,[256] ,[0 ,256])
	axarr[1 ,0].plot(hist_orig, color = c)
	hist_gamma = cv.calcHist([img_gamma] ,[i] ,None ,[256] ,[0 ,256])
	axarr[1 ,0].plot(hist_gamma, color = c)

fig,ax = plt.subplots()
axarr [2, 0].plot(table)
axarr [2, 0].set_xlim(0,255)
axarr [2, 0].set_ylim(0,255)
axarr [2, 0].set_aspect('equal')
plt.show()