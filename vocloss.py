import tensorflow as tf
from glob import glob
import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy

class CrossEntropy(Loss):
    def __init__(self, ignored_index = 255):
        super(CrossEntropy, self).__init__()
        self.ignored_index = ignored_index
        self.entropy = SparseCategoricalCrossentropy()

    def call(self, y_true, y_pred):
        """
        Cross entropy loss function with ignore index
        :param y_true -- labels -- b, h, w
        :param y_pred -- probability tensor -- b, h, w, c
        :return:
        """
        C = y_pred.shape[3]
        y_pred = tf.reshape(y_pred, (-1, C))
        y_true = tf.reshape(y_true, (-1,))

        valid = tf.not_equal(y_true, self.ignored_index)
        vy_pred = tf.boolean_mask(y_pred, valid)
        vy_true = tf.boolean_mask(y_true, valid)

        loss = self.entropy(vy_true, vy_pred)

        return loss





IMAGES_PATH = "/home/minouei/Downloads/datasets/VOCdevkit/VOC2012/train/"
# MASKS_PATH = "/home/minouei/Downloads/datasets/VOCdevkit/VOC2012/SegmentationClass/"
VAL_PATH = "/home/minouei/Downloads/datasets/VOCdevkit/VOC2012/minival/"
IMAGE_FILE = "/home/minouei/Downloads/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt"
VAL_FILE = "/home/minouei/Downloads/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/minival.txt"
NumClasses = 22
BATCH_SIZE = 4
RESTORE_CHECKPOINT = True
BUFFER_SIZE = 1
IMG_HEIGHT = 512
IMG_WIDTH = 512
img_h, img_w = 512,512
EPOCHS = 20
START = 0



@tf.function
def read_img(image_path, mask=False):
    img = tf.io.read_file(image_path)
    if mask:
        # input_mask = Image.open (image_path).convert ("P").resize ((IMG_HEIGHT, IMG_WIDTH))
        # input_mask = np.array (input_mask)
        # img = np.where (input_mask>NumClasses, 0, input_mask)

        img = tf.image.decode_png(img, channels=1)
        img = tf.where (tf.equal(img,255), tf.cast(21, tf.uint8), img)
        img.set_shape([None, None, 1])
        img = (tf.image.resize(images=img, size=[
            img_h, img_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
        # img = tf.cast(img, tf.float32)
    else:
        img = tf.image.decode_png(img, channels=3)
        img.set_shape([None, None, 3])
        img = (tf.image.resize(images=img, size=[
            img_h, img_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
        img = tf.cast(img, tf.float32) / 255.0
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
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


train_images_folder = sorted(glob(os.path.join(IMAGES_PATH, 'img/*.jpg')))
train_mask_folder = sorted(glob(os.path.join(IMAGES_PATH, 'mask/*.png')))
valid_images_folder = sorted(glob(os.path.join(VAL_PATH, 'img/*.jpg')))
valid_mask_folder = sorted(glob(os.path.join(VAL_PATH, 'mask/*.png')))

train_dataset = data_generator(train_images_folder, train_mask_folder)
val_dataset = data_generator(valid_images_folder, valid_mask_folder)




def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    return pred_mask[0]

def display(display_list,ii):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']
  for i in range(len(display_list)):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    plt.imshow(display_list[i])
    plt.axis('off')
  plt.savefig(str(ii)+'200dpi.png', dpi=200)

def show_predictions(model,dataset=None,sample_image=None, sample_mask=None, num=1):
    if dataset:
        ii=1
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            print(create_mask(pred_mask).shape)
            display([image[0], mask[0], create_mask(pred_mask)],ii)
            ii=ii+1
    else:
        display([sample_image, sample_mask,
            create_mask(model.predict(sample_image[tf.newaxis, ...]))])
        
class SparseMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self,
                 num_classes=None,
                 name='mean_iou',
                 dtype=None):
        super(SparseMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)  # noqa: E501
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(SparseMeanIoU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# model = tf.keras.models.load_model('trained/deeplab_first',compile=False)
# show_predictions(model,val_dataset, num = 20)
# exit()


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay (initial_learning_rate=1e-4 * (0.96 ** START), decay_steps=50000, decay_rate=0.96, staircase=True)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# loss=tf.keras.losses.SparseCategoricalCrossentropy()

loss=CrossEntropy(21)
# loss=my_loss

savedModel=("deeplab_512")
baseModel = tf.keras.models.load_model(savedModel,custom_objects={'SparseMeanIoU': SparseMeanIoU})
baseModel.summary()
output = tf.keras.layers.Conv2D(NumClasses,(1,1),activation="softmax",padding="same")(baseModel.layers[-2].output)
model = tf.keras.Model(baseModel.layers[0].input,output)




model.summary()
model.compile(loss=loss, optimizer=opt,     metrics=["accuracy"])

checkpoint_filepath = 'checkpoints'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
    
model_history = model.fit(train_dataset, epochs=EPOCHS, 
                          callbacks=[model_checkpoint_callback],
                          validation_data=val_dataset)


write_model_path= 'trained/deeplab_first'
model.save(write_model_path)

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.savefig('loss.png')



show_predictions(model,val_dataset, num = 20)














