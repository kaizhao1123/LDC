import os

from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import numpy as np
import matplotlib.pyplot as plt


def myFunc(image):
    image = np.array(image)
    hsv_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    return Image.fromarray(hsv_image)
    # return hsv_image


def augmentOneImage(resourse, imageName, saveDir, saveType, seed):
    datagen = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.5, 1.5],
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        # cval=0,
        fill_mode='reflect')

    img_ori = load_img('C:/Users/Kai Zhao/PycharmProjects/LDC/' + resourse + '/' + imageName)  # this is a PIL image
    imageNum = imageName.split('.')[0]
    x_ori = img_to_array(img_ori)  # this is a Numpy array with shape (3, 150, 150)
    x_ori = x_ori.reshape((1,) + x_ori.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    i = 0
    for batch in datagen.flow(x_ori, batch_size=1,
                              save_to_dir='C:/Users/Kai Zhao/PycharmProjects/LDC/augResult/' + saveDir,
                              save_prefix='aug_'+imageNum,
                              save_format=saveType, seed=seed):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely


def augmentBoth():

    seed = 1
    data_dir = 'C:/Users/Kai Zhao/PycharmProjects/LDC/'
    image_directories = data_dir + 'data'

    for imageName in os.listdir(image_directories):
        imageNum = os.path.splitext(imageName)[0]
        augmentOneImage('data', imageName, 'img', 'bmp', seed)
        augmentOneImage('data_gt', imageNum+'.jpg', 'edge', 'jpg', seed)
        seed+=1

    save_path = os.path.join(data_dir+'augResult', 'train_pair.lst')


def augmentBoth_test():
    # datagen = ImageDataGenerator(
    #     rotation_range=0.2,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     brightness_range=[0.5, 1.5],
    #     rescale=1. / 255,
    #     shear_range=0.2,
    #     zoom_range=0.3,
    #     horizontal_flip=True,
    #     fill_mode='nearest')
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rescale=1. / 255,
                         rotation_range=90,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         zoom_range=0.3,
                         # horizontal_flip=True,
                         # shear_range=0.2,
                         # fill_mode='nearest',
                         # brightness_range=[0.5, 1.5]
                         )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1
    image_generator = image_datagen.flow_from_directory(
        # 'C:/Users/Kai Zhao/PycharmProjects/LDC/dataset/Milo/edges/imgs/train/rgbr/real',
        'C:/Users/Kai Zhao/PycharmProjects/LDC/data',
        batch_size=1,
        target_size=(256, 280),
        class_mode=None,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        # 'C:/Users/Kai Zhao/PycharmProjects/LDC/dataset/Milo/edges/edge_maps/train/rgbr/real',
        'C:/Users/Kai Zhao/PycharmProjects/LDC/data_gt',
        batch_size=1,
        target_size=(256, 280),
        class_mode=None,
        seed=seed)
    print(image_generator)

    train_generator = zip(image_generator, mask_generator)

    i = 0
    for x_image, x_mask in train_generator:

        # for j in range(5):
        #     plt.subplot(2,5,j+1)
        #     plt.imshow(x_image[i])
        #     plt.subplot(2,5,5+j+j)
        #     plt.imshow(x_mask)
        # plt.show()

        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely


if __name__ == '__main__':
    # augmentOneImage()
    augmentBoth()
