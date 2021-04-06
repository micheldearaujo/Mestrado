import rasterio
from rasterio.plot import show
path = 'D:/michel/data/amazonia/tif/'
file = 'Hansen_GFC-2019-v1.7_first_00N_070W.tif'

img = rasterio.open(path+file)
show(img)