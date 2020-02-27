import tensorflow as tf
import numpy as np
import os
import sys
import gflags
import time
from keras.callbacks import ModelCheckpoint
from keras import optimizers

import logz
import cnn_models
import utils
import log_utils
from common_flags import FLAGS
from keras.callbacks import TensorBoard


def getModel(img_width, img_height, img_channels, output_dim, weights_path):
    """
    Initialize model.

    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.
       weights_path: Path to pre-trained model.

    # Returns
       model: A Model instance.
    """
    model = cnn_models.resnet8_MDN(img_width, img_height, img_channels, output_dim)

    if weights_path:
        try:
            model.load_weights(weights_path)
            print("Loaded model from {}".format(weights_path))
        except:
            print("Impossible to find weight path. Returning untrained model")

    return model


def trainModel(train_data_generator, val_data_generator, model, initial_epoch):
    """
    Model training.

    # Arguments
       train_data_generator: Training data generated batch by batch.
       val_data_generator: Validation data generated batch by batch.
       model: Target image channels.
       initial_epoch: Dimension of model output.
    """

    # Initialize loss weights
    model.alpha = tf.Variable(1, trainable=False, name='alpha', dtype=tf.float32)
    model.beta = tf.Variable(1, trainable=False, name='betabeta', dtype=tf.float32)

    # Initialize number of samples for hard-mining
    model.k_mse = tf.Variable(FLAGS.batch_size, trainable=False, name='k_mse', dtype=tf.int32)
    model.k_entropy = tf.Variable(FLAGS.batch_size, trainable=False, name='k_entropy', dtype=tf.int32)


    # optimizer = optimizers.Adam(decay=1e-5)
    optimizer = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    # optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
    # Configure training process
    # model.compile(loss=[utils.mean_log_Gaussian_like_with_sigma_supress,
    #                    utils.hard_mining_mse_translation(model.k_mse)],
    #                    optimizer=optimizer, loss_weights=[model.alpha, model.beta])

    model.compile(loss=[utils.mean_log_Gaussian_like_with_sigma_supress,
                        utils.hard_mining_mse_translation(model.k_mse)],
                  optimizer=optimizer, loss_weights=[model.alpha, model.beta])

    s_time = time.strftime("%Y%m%d%H%M%S", time.localtime())  # 时间戳
    # logs文件路径
    logs_path = './logs/log_%s' % (s_time)

    try:
        os.makedirs(logs_path)
    except:
        pass

    # 将loss ，acc， val_loss ,val_acc记录tensorboard
    tensorboard = TensorBoard(log_dir=logs_path, write_graph=True)

    # Save model with the lowest validation loss
    weights_path = os.path.join(FLAGS.experiment_rootdir, 'weights_{epoch:03d}.h5')
    writeBestModel = ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                                     save_best_only=True, save_weights_only=True)

    # Save model every 'log_rate' epochs.
    # Save training and validation losses.
    logz.configure_output_dir(FLAGS.experiment_rootdir)
    saveModelAndLoss = log_utils.MyCallback(filepath=FLAGS.experiment_rootdir,
                                            period=FLAGS.log_rate,
                                            batch_size=FLAGS.batch_size)

    # Train model
    steps_per_epoch = int(np.ceil(train_data_generator.samples / FLAGS.batch_size))
    validation_steps = int(np.ceil(val_data_generator.samples / FLAGS.batch_size))

    model.fit_generator(train_data_generator,
                        epochs=FLAGS.epochs, steps_per_epoch = steps_per_epoch,
                        callbacks=[writeBestModel, saveModelAndLoss, tensorboard],
                        validation_data=val_data_generator,
                        validation_steps = validation_steps,
                        initial_epoch=initial_epoch)


def _main():

    # Create the experiment rootdir if not already there
    if not os.path.exists(FLAGS.experiment_rootdir):
        os.makedirs(FLAGS.experiment_rootdir)

    # Input image dimensions
    img_width, img_height = FLAGS.img_width, FLAGS.img_height

    # Cropped image dimensions
    crop_img_width, crop_img_height = FLAGS.crop_img_width, FLAGS.crop_img_height

    # Image mode
    if FLAGS.img_mode=='rgb':
        img_channels = 3
    elif FLAGS.img_mode == 'grayscale':
        img_channels = 1
    else:
        raise IOError("Unidentified image mode: use 'grayscale' or 'rgb'")

    # Output dimension (one for steering and one for collision)
    output_dim = 1

    # Generate training data with real-time augmentation
    train_datagen = utils.DroneDataGenerator(rescale = 1./255
                                            # ,zoom_range = [0.7,1]
                                             )

    train_generator = train_datagen.flow_from_directory(FLAGS.train_dir,
                                                        shuffle = True,
                                                        color_mode=FLAGS.img_mode,
                                                        target_size=(img_height, img_width),
                                                        crop_size=(crop_img_height, crop_img_width),
                                                        batch_size = FLAGS.batch_size)

    # Generate validation data with real-time augmentation
    val_datagen = utils.DroneDataGenerator(rescale = 1./255)

    val_generator = val_datagen.flow_from_directory(FLAGS.val_dir,
                                                        shuffle = True,
                                                        color_mode=FLAGS.img_mode,
                                                        target_size=(img_height, img_width),
                                                        crop_size=(crop_img_height, crop_img_width),
                                                        batch_size = FLAGS.batch_size)

    # Weights to restore
    weights_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.weights_fname)
    initial_epoch = 0
    if not FLAGS.restore_model:
        # In this case weights will start from random
        weights_path = None
    else:
        # In this case weigths will start from the specified model
        initial_epoch = FLAGS.initial_epoch


    # Define model
    model = getModel(crop_img_width, crop_img_height, img_channels,
                        output_dim, weights_path)

    # Serialize model into json
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    utils.modelToJson(model, json_model_path)

    # Train model
    trainModel(train_generator, val_generator, model, initial_epoch)


def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))

      sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
