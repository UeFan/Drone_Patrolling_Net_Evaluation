import gflags



FLAGS = gflags.FLAGS

# Input
# The image will be first resized to this imgSize
gflags.DEFINE_integer('img_width', 400, 'Target Image Width')
gflags.DEFINE_integer('img_height', 225, 'Target Image Height')

# Then the image will be cropped to this crop_size
gflags.DEFINE_integer('crop_img_width', 400, 'Cropped image widht')
gflags.DEFINE_integer('crop_img_height', 100, 'Cropped image height')
gflags.DEFINE_integer('crop_img_width_res18', 320, 'Cropped image widht of resnet')
gflags.DEFINE_integer('crop_img_height_res18', 180, 'Cropped image height of resnet')
gflags.DEFINE_integer('crop_img_width_dronet', 200, 'Cropped image widht of dronet')
gflags.DEFINE_integer('crop_img_height_dronet', 200, 'Cropped image height of dronet')
gflags.DEFINE_string('img_mode', "rgb", 'Load mode for images, either '
                     'rgb or grayscale')

# Training
gflags.DEFINE_integer('batch_size', 5, 'Batch size in training and evaluation')
gflags.DEFINE_integer('epochs', 250, 'Number of epochs for training')
gflags.DEFINE_integer('log_rate', 10, 'Logging rate for full model (epochs)')
gflags.DEFINE_integer('initial_epoch', 0, 'Initial epoch to start training')

# MDN output
gflags.DEFINE_integer('output_dimension', 1, 'The number of outputs we want to predict')
gflags.DEFINE_integer('distribution_num', 3, 'The number of distributions we want to use in the mixture')

# Files
gflags.DEFINE_string('experiment_rootdir', "./model", 'Folder '
                     ' containing all the logs, model weights and results')
gflags.DEFINE_string('train_dir', "/home/rikka/uav-project/drone-data-train", 'Folder containing'
                     ' training experiments')
gflags.DEFINE_string('val_dir', "/home/rikka/uav-project/drone-data-validation", 'Folder containing'
                     ' validation experiments')
gflags.DEFINE_string('test_dir', "/home/rikka/uav-project/drone-data-train", 'Folder containing'
                     ' testing experiments')
gflags.DEFINE_string('model_dir', "/home/rikka/uav-project/drone-data-train", 'Folder containing'
                     ' trained model')
# Model
gflags.DEFINE_bool('restore_model', False, 'Whether to restore a trained'
                   ' model for training')
gflags.DEFINE_string('weights_fname', "model_weights.h5", '(Relative) '
                                          'filename of model weights')
gflags.DEFINE_string('json_model_fname', "model_struct.json",
                          'Model struct json serialization, filename')

