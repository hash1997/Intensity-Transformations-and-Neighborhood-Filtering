import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#generating intensity map
trf = np.linspace(start = 0, stop = 50, num = 51, dtype=np.uint8)
trf = np.insert(trf,trf.size, np.linspace(start = 100, stop = 255, num = 100, dtype=np.uint8))
trf = np.insert(trf,trf.size-1, np.linspace(start = 150, stop = 255, num = 105, dtype=np.uint8))

fig,ax = plt.subplots()
ax.plot(trf)
ax.set_xlabel( r'Input_Intensity, $f(\mathbf{x})$')
ax.set_ylabel('Output_Intensity, $\mathrm{T}[f(\mathbf{x})]$')
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.set_aspect('equal')
plt.show()

#loading the image
img_loaded = cv.imread('emma_gray.jpg', cv.IMREAD_GRAYSCALE)
img_transformed = cv.LUT(img_loaded, trf)
fig,ax = plt.subplots(1,2,figsize=(5,15))

ax[0].imshow(img_loaded,cmap='gray', vmin=0, vmax=255)
ax[1].imshow(img_transformed,cmap='gray', vmin=0, vmax=255)
plt.show()