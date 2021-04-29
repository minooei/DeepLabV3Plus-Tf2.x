import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
import os

img_w, img_h = 704, 704
batch_size = 1
num_classes = 19
DATA_DIR = "/home/minouei/Downloads/datasets/contract/version5"


@tf.function
def read_img(image_path, mask=False):
    img = tf.io.read_file(image_path)
    if mask:
        img = tf.image.decode_png(img, channels=1)
        img.set_shape([None, None, 1])
        img = (tf.image.resize(images=img, size=[
            img_h, img_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
        img = tf.cast(img, tf.float32)
    else:
        img = tf.image.decode_png(img, channels=3)
        img.set_shape([None, None, 3])
        img = (tf.image.resize(images=img, size=[
            img_h, img_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
        img = tf.cast(img, tf.float32) / 127.5 - 1
    return img


def load_data(img_list, mask_list):
    img = read_img(img_list)
    mask = read_img(mask_list, mask=True)
    return img, mask


def data_generator(img_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((img_list,
                                                  mask_list))
    dataset = dataset.map(
        load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


valid_images_folder = sorted(glob(os.path.join(DATA_DIR, 'images/val/*.jpg')))
valid_mask_folder = sorted(glob(os.path.join(DATA_DIR, 'annotations/val/*.png')))

valid_dataset = data_generator(valid_images_folder, valid_mask_folder)

from focal_loss import SparseCategoricalFocalLoss
model = tf.keras.models.load_model('trained/deeplab_final',custom_objects={'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss})


from PIL import Image

def inference(dataset):
    def OverLayLabelOnImage(ImgIn, Label, W=0.6):
        # ImageIn is the image
        # Label is the label per pixel
        # W is the relative weight in which the labels will be marked on the image
        # Return image with labels marked over it
        Img = ImgIn.copy()
        TR = [0, 1, 0, 0,   0, 1, 1, 0, 0,   0.5, 0.7, 0.3, 0.5, 1,    0.5, 0.3]
        TB = [0, 0, 1, 0,   1, 0, 1, 0, 0.5, 0,   0.2, 0.2, 0.7, 0.5,  1,   0.3]
        TG = [0, 0, 0, 0.5, 1, 1, 0, 1, 0.7, 0.4, 0.7, 0.2, 0,   0.25, 0.5, 0.3]
        R = Img[:, :, 0].copy()
        G = Img[:, :, 1].copy()
        B = Img[:, :, 2].copy()
        for i in range(Label.max()+1):
            if i < len(TR):  # Load color from Table
                R[Label == i] = TR[i] * 255
                G[Label == i] = TG[i] * 255
                B[Label == i] = TB[i] * 255
            else:  # Generate random label color
                R[Label == i] = np.mod(i*i+4*i+5, 255)
                G[Label == i] = np.mod(i*10, 255)
                B[Label == i] = np.mod(i*i*i+7*i*i+3*i+30, 255)
        Img[:, :, 0] = Img[:, :, 0] * (1 - W) + R * W
        Img[:, :, 1] = Img[:, :, 1] * (1 - W) + G * W
        Img[:, :, 2] = Img[:, :, 2] * (1 - W) + B * W
        return Img

    for j,val in enumerate(dataset.take(11)):
        predsTrain = model.predict(np.expand_dims((val[0][0]), axis=0))
        out = np.squeeze(predsTrain)
        y = np.argmax(out, axis=2)
        img = tf.cast((val[0][0]+1)*127.5, tf.uint8)
        out=OverLayLabelOnImage(img.numpy(),y)
        im = Image.fromarray(out)
        im.save( str(j)+'.png')


inference(valid_dataset)
