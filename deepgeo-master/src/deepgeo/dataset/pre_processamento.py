import deepgeo.dataset.preprocessor as prep

preproc = prep.Preprocessor('my_raster.tif', no_data=0)

def ndwi(raster, param):
    nir = raster[:,:,param['idx_b_nir']]
    swir = raster[:,:,param['idx_b_swir']]
    ndwi = (nir - swir)/(nir+swir)
    return ndwi

self.preproc.register_new_idx_fun('ndwi', ndwi)

preproc.compute_indeces({
    'ndvi':{'idx_b_red':3, 'idx_b_nir':4},
    'ndwi':{'idx_bswir':5, 'idx_b_nir':4}})

preproc.standardize_image()
preproc.save_stacked_raster('output.tif')