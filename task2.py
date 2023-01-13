import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img_loaded = cv.imread('brain_proton_density_slice.png', cv.IMREAD_GRAYSCALE)

#histogram 
hist = cv.calcHist([img_loaded], [0], None, [256], [0,256])
plt.plot(hist)
plt.xlim([0,256])
plt.show()

trf = np.linspace(start = 0, stop = 150, num = 150, dtype=np.uint8)
trf = np.insert(trf,trf.size, np.linspace(start = 150, stop = 155, num = 30, dtype=np.uint8))
trf = np.insert(trf,trf.size, np.linspace(start = 200, stop = 255, num = 76, dtype=np.uint8))

fig,ax = plt.subplots()
ax.plot(trf)
ax.set_xlabel( r'Input_Intensity, $f(\mathbf{x})$')
ax.set_ylabel('Output_Intensity, $\mathrm{T}[f(\mathbf{x})]$')
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.set_aspect('equal')
plt.show()


#Gaussian filtering before transformation
img_loaded_blur = cv.GaussianBlur(img_loaded,(5,5),0)

img_transformed = cv.LUT(img_loaded_blur, trf)
fig,ax = plt.subplots(1,2,figsize=(5,15))
ax[0].imshow(img_loaded,cmap='gray', vmin=0, vmax=255)
ax[1].imshow(img_transformed,cmap='gray', vmin=0, vmax=255)
plt.show()