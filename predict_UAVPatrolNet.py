import os,sys
import utils
# import merge_models
import cv2
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from common_flags import FLAGS
import socket
import time
from keras import backend as K
TEST_PHASE=0
import cnn_models
#def listen_to_map(a):


def central_image_crop(img, crop_height,crop_width):
    half_the_width = int(img.shape[1] / 2)

    img = img[(img.shape[0] - crop_height): img.shape[0],
          int(half_the_width - (crop_width / 2)): int(half_the_width + (crop_width / 2))]
    if FLAGS.img_mode == 'grayscale':
        img = img.reshape((img.shape[0], img.shape[1], 1))
    return img

def sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    without_max = np.exp(x) / np.sum(np.exp(x))
    x_max = np.max(x, axis=axis, keepdims=True)
    with_max = np.exp(x - x_max) / np.sum(np.exp(x - x_max))
    #print(without_max)
    print(with_max)
    return with_max

def gaussian(sigs, mus, pis, x):
    gmm = 0
    for sigma, u, pi in zip(sigs, mus, pis):
        pi = pi / 2
        y = np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
        # print(sigma,u,pi,x,y)
        gmm = gmm + y * pi
    return gmm



def main():


    json_model_path = '/model/UAVPatrolNet_model_struct.json'
    weights_path = FLAGS.model_dir
    pics_path = FLAGS.test_dir

    # Input image dimensions
    img_width, img_height = FLAGS.img_width, FLAGS.img_height

    # Cropped image dimensions
    crop_img_width, crop_img_height = FLAGS.crop_img_width, FLAGS.crop_img_height
    target_size = (img_height,img_width)
    crop_size = (crop_img_height,crop_img_width)

    # Set keras utils
    K.set_learning_phase(TEST_PHASE)
    # Load json and create model
    model = utils.jsonToModel(json_model_path)
#     model = cnn_models.resnet8_MDN(crop_img_width,crop_img_height,3,1)
    #model = merge_models.merge_model()
    # Load weights
    model.load_weights(weights_path,by_name=True)
    #model.compile(loss='mse', optimizer='sgd')
    model.compile(loss='mse', optimizer='adam')

    print("json_model_path: {}".format(json_model_path))
    print("Loaded model from {}".format(weights_path))
    #cv2.namedWindow("img", 0);
    #cv2.resizeWindow("img", 640, 360);
    #cv2.namedWindow("crop",0);
    #cv2.resizeWindow("crop", 400,160);

    pic_list = os.listdir(pics_path)
    pic_list.sort()
    counter = 0

    roll_speed = 0.
    try:
        while True:
            time_start=time.time()

            pic = pic_list[int(counter)]
            counter = counter + 1
            print(pic)
            img_origi = cv2.imread(os.path.join(pics_path, pic), cv2.IMREAD_COLOR)
            img_origi = cv2.resize(img_origi, (640, 360))
            # run predict
            if FLAGS.img_mode == 'grayscale':
                img = cv2.cvtColor(img_origi, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (target_size[1], target_size[0]))
            else:
                img = cv2.resize(img_origi, (target_size[1], target_size[0]))

            img = central_image_crop(img, crop_size[0], crop_size[1])
            if FLAGS.img_mode == 'grayscale':
                img = img.reshape((img.shape[0], img.shape[1], 1))
            cv_image = np.asarray(img, dtype=np.float32) * np.float32(1.0/255.0)
            # print(cv_image)
            outs = model.predict_on_batch(cv_image[None])
            #print(len(outs[0]))
            parameter, translation = outs[0][0], outs[1][0]
            #print("steer = {}, translation = {}".format(parameter,translation))

            y_pred = np.reshape(parameter, [-1, 6])
            out_mu, out_pi = np.split(y_pred, 2, axis=1)
            #print(out_pi)
            pi = sum_exp(out_pi, 1)
            pi = np.split(pi, 3, axis=1)
            # component_splits = [1, 1, 1]
            mus = np.split(out_mu, 3, axis=1)

            out_sigma = np.array([[0.1, 0.1, 0.1]], dtype='float32')
            sigs = np.split(out_sigma, 3, axis=1)

            x = np.linspace(-1, 1, 100)
            y = np.array([])
            for x_ in x:
                y = np.append(y, gaussian(sigs, mus, pi, x_))


            possible_direct = []
            possible_roll_speed = []

            start = 0
            continue_flag = 0
            sum_y = 0
            sum_x = 0
            for x_, y_ in zip(x, y):
                # print(point)
                if(y_ > 0.3):
                    if(continue_flag == 0):
                        continue_flag = 1
                        start = x_
                    sum_y = sum_y + y_
                    sum_x = sum_x + 1
                    y_ = (img_origi.shape[0] - y_ * 200 - 80).astype(np.int32)
                    x_ = ((x_ + 1) / 2 * img_origi.shape[1]).astype(np.int32)
                    x_ = img_origi.shape[1] - x_
                    cv2.circle(img_origi, (x_, int(y_/2)+150), 3, (0, 255, 0), 4)
                else:
                    if(continue_flag == 1):
                        continue_flag = 0
                        possible_direct.append((x_ + start)/2)
                        possible_roll_speed.append((sum_y/sum_x - 1.)/2)
                        sum_y = 0
                        sum_x = 0
                    y_ = (img_origi.shape[0] - y_ * 200 - 80).astype(np.int32)
                    x_ = ((x_ + 1) / 2 * img_origi.shape[1]).astype(np.int32)
                    x_ = img_origi.shape[1] - x_
                    cv2.circle(img_origi, (x_, int(y_/2)+150), 1, (255, 0, 255), 4)
#            print("====Map_direct = {} ====".format(map_direct))
            map_direct = 0
            min_direct_diff = 180
            steer = 0.
            roll_speed_ = 0
            count = 0
            for possible_direct_ in possible_direct:
                # print(possible_direct_)
                cv2.line(img_origi, (int(img_origi.shape[1] / 2), img_origi.shape[0] - 50),
                     (int(img_origi.shape[1] / 2 - math.tan(possible_direct_ * 3.14 / 2) * 100), img_origi.shape[0] - 150),
                     (0, 255, 0), 3)
                diff = abs(-possible_direct_*90 - map_direct)
                if(diff<min_direct_diff):
                    min_direct_diff = diff
                    steer = possible_direct_
                    roll_speed_ = possible_roll_speed[count]
                count = count + 1

            cv2.line(img_origi, (int(img_origi.shape[1] / 2), img_origi.shape[0] - 50),
                     (int(img_origi.shape[1] / 2 - math.tan(steer * 3.14 / 2) * 100), img_origi.shape[0] - 150),(0, 255, 255), 3)
            # map_direct = map_direct/90
            # seq = "ab"+'%f'%(map_direct*400)+',%f'%(0*200)

            roll_speed = roll_speed*0.9 + roll_speed_*0.1



            # cv2.line(img_origi, (int(img_origi.shape[1]/2),img_origi.shape[0]), (int(img_origi.shape[1]/2),50), (0,255,0), 1)
            cv2.line(img_origi, (int(img_origi.shape[1]/2),img_origi.shape[0]-50), (int((translation+1)/2*img_origi.shape[1]), img_origi.shape[0] - 50), (255,255,0), 8)
            cv2.imshow("img", img_origi)
            cv2.imshow('crop',img)
            #cv2.imwrite(pics_path + '_save        /our' + pic, img_origi)
            time_end=time.time()
            print('totally cost',time_end-time_start)
            cv2.waitKey(0)
    except KeyboardInterrupt:

        print("calling to end")
if __name__ == '__main__':
    main()