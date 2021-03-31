import deepgeo.dataset.dataset_generator as dsgen

dataset_description = {'standardization':'norm_range',
                       'spectr_indexes':['ndvi','ndwi'],
                       'sensor':'Landsat-8 OLI',
                       'classes':['deforestation','forest'],
                       'img_no_data':0,
                       'chip_size':316,
                       'notes':'Dataset for an example'}

raster_array = prep.get_array_stacked_raster()
labels_array = rasterizer.get_labeled_raster()
generator = dsgen.DatasetGenerator(raster_array, labels_array,
                                   description=dataset_description)

generator.generate_chips(params={'win_size':316})
generator.remove_no_data(tolerance=0.5)
generator.shuffle_ds()
generator.split_ds()
generator.save_to_disk(out_path='./my_dataset', filename='dataset')