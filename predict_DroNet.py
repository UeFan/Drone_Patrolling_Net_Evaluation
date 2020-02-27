import os,sys
import numpy as np
from common_flags import FLAGS
TEST_PHASE=0
import cnn_models
import img_utils
import cv2
import math
from predict_UAVPatrolNet import gaussian, sum_exp
from keras.utils import plot_model
def main():

    weights_path = FLAGS.model_dir
    pics_path = FLAGS.test_dir

    # Input image dimensions
    img_width, img_height = FLAGS.img_width, FLAGS.img_height
    # Cropped image dimensions
    crop_img_width, crop_img_height = FLAGS.crop_img_width_dronet, FLAGS.crop_img_height_dronet
    target_size = (img_height,img_width)
    crop_size = (crop_img_height,crop_img_width)

    model = cnn_models.resnet8(crop_img_width,crop_img_height,1,1)
    plot_model(model,"model.png")
    model.load_weights(weights_path,by_name=True)
    model.compile(loss='mse', optimizer='adam')
    pic_list = os.listdir(pics_path)
    pic_list.sort()
    try:
        for img_name in pic_list:
            current_name = pics_path + '/' + img_name
            img = img_utils.load_img(current_name,grayscale=True,target_size = target_size,crop_size = crop_size)
            img = np.asarray(img, dtype=np.float32) * np.float32(1.0 / 255.0)
            outs = model.predict_on_batch(img[None])
            steer, coll = outs[0][0], outs[1][0]
            steer = np.max(steer)
            coll = 0 * coll
            img_origi = img_utils.load_img(current_name,target_size=(320, 640))
            out_mu = np.array([[0,0,steer]])
            out_pi = np.array([[0,0,1]])
            # print(out_pi)
            # pi = sum_exp(out_pi, 1)
            pi = np.split(out_pi, 3, axis=1)
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
                if (y_ > 0.3):
                    if (continue_flag == 0):
                        continue_flag = 1
                        start = x_
                    sum_y = sum_y + y_
                    sum_x = sum_x + 1
                    y_ = (img_origi.shape[0] - y_ * 200 - 80).astype(np.int32)
                    x_ = ((x_ + 1) / 2 * img_origi.shape[1]).astype(np.int32)
                    x_ = img_origi.shape[1] - x_
                    #cv2.circle(img_origi, (x_, int(y_ / 2) + 150), 3, (0, 255, 0), 4)
                else:
                    if (continue_flag == 1):
                        continue_flag = 0
                        possible_direct.append((x_ + start) / 2)
                        possible_roll_speed.append((sum_y / sum_x - 1.) / 2)
                        sum_y = 0
                        sum_x = 0
                    y_ = (img_origi.shape[0] - y_ * 200 - 80).astype(np.int32)
                    x_ = ((x_ + 1) / 2 * img_origi.shape[1]).astype(np.int32)
                    x_ = img_origi.shape[1] - x_
                    #cv2.circle(img_origi, (x_, int(y_ / 2) + 150), 1, (255, 0, 255), 4)
            #            print("====Map_direct = {} ====".format(map_direct))
            map_direct = 0
            min_direct_diff = 180
            steer = 0.
            roll_speed_ = 0
            count = 0
            for possible_direct_ in possible_direct:
                # print(possible_direct_)
                cv2.line(img_origi, (int(img_origi.shape[1] / 2), img_origi.shape[0] - 50),
                         (int(img_origi.shape[1] / 2 - math.tan(possible_direct_ * 3.14 / 2) * 100),
                          img_origi.shape[0] - 150),
                         (0, 255, 0), 3)
                diff = abs(-possible_direct_ * 90 - map_direct)
                if (diff < min_direct_diff):
                    min_direct_diff = diff
                    steer = possible_direct_
                    roll_speed_ = possible_roll_speed[count]
                count = count + 1

            cv2.line(img_origi, (int(img_origi.shape[1] / 2), img_origi.shape[0] - 50),
                     (int(img_origi.shape[1] / 2 - math.tan(steer * 3.14 / 2) * 100), img_origi.shape[0] - 150),
                     (0, 255, 255), 3)
            cv2.imshow("img", img_origi / 255)
            #cv2.imwrite(pics_path + '_save/dronet_' + img_name,img_origi)
            cv2.waitKey(10)
    except KeyboardInterrupt:
        print("calling to end")
if __name__ == '__main__':
    main()