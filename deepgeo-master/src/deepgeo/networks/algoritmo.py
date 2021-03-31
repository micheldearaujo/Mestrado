import deepgeo.networks.model_builder as mb
import deepgeo.dataset.utils as dsutils

model_dir = 'trained_model'
params = {'network':'fcn8s',
          'epochs':100,
          'batch_size':20,
          'learning_rate':0.1,
          '12_reg_rate':0.0005,
          'data_aug_ops':['rot90','rot180','rot270','flip_left_right',
                          'flip_up_down','flip_transpose']}

model=mb.ModelBuilder(params)
model.train('dataset_train.tfrecord', 'dataset_test.tfrecord',
            model_dir)

model.validate('dataset_valid.tfrecord', model_dir)
model.predict('img.tif', model_dir, 'classif.tif')