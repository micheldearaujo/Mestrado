import deepgeo.dataset.rasterizer as rast

class_column = 'class'

classes_of_inter=['deforestation','forest']

rasterizer = rast.Rasterizer(vector_file='my_labels.shp',
                             in_raster_file='my_raster.tiff',
                             class_column = class_column,
                             classes_interest=classes_of_inter)

rasterizer.rasterize_layer()
rasterizer.save_labeled_raster_to_gtiff('my_labels.tif')