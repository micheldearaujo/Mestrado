from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

Image.MAX_IMAGE_PIXELS = None

im = Image.open('D:/michel/data/amazonia/Hansen_GFC-2019-v1.7_treecover2000_00N_070W.tif')

im.show()
im_array = np.array(im)
print(im_array.shape)
print(im.size)

#I = plt.imread('D:/michel/data/amazonia/Hansen_GFC-2019-v1.7_treecover2000_00N_070W.tif')