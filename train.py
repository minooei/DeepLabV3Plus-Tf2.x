import tensorflow as tf
from glob import glob
import os
from focal_loss import SparseCategoricalFocalLoss

img_w, img_h = 704, 704
batch_size = 4
num_classes = 19
epochs = 6


DATA_DIR = "/home/minouei/Downloads/datasets/contract/version5"
# DATA_DIR = "/netscratch/minouei/versicherung/version5"


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


train_images_folder = sorted(glob(os.path.join(DATA_DIR, 'images/train/*.jpg')))
train_mask_folder = sorted(glob(os.path.join(DATA_DIR, 'annotations/train/*.png')))
valid_images_folder = sorted(glob(os.path.join(DATA_DIR, 'images/val/*.jpg')))
valid_mask_folder = sorted(glob(os.path.join(DATA_DIR, 'annotations/val/*.png')))

train_dataset = data_generator(train_images_folder, train_mask_folder)
valid_dataset = data_generator(valid_images_folder, valid_mask_folder)


def get_model():
    def AtrousSpatialPyramidPooling(model_input):
        dims = tf.keras.backend.int_shape(model_input)

        layer = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3],
                                                            dims[-2]))(model_input)
        layer = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same',
                                       kernel_initializer='he_normal')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)
        out_pool = tf.keras.layers.UpSampling2D(size=(dims[-3] // layer.shape[1],
                                                      dims[-2] // layer.shape[2]),
                                                interpolation='bilinear')(layer)

        layer = tf.keras.layers.Conv2D(256, kernel_size=1,
                                       dilation_rate=1, padding='same',
                                       kernel_initializer='he_normal',
                                       use_bias=False)(model_input)
        layer = tf.keras.layers.BatchNormalization()(layer)
        out_1 = tf.keras.layers.ReLU()(layer)

        layer = tf.keras.layers.Conv2D(256, kernel_size=3,
                                       dilation_rate=6, padding='same',
                                       kernel_initializer='he_normal',
                                       use_bias=False)(model_input)
        layer = tf.keras.layers.BatchNormalization()(layer)
        out_6 = tf.keras.layers.ReLU()(layer)

        layer = tf.keras.layers.Conv2D(256, kernel_size=3,
                                       dilation_rate=12, padding='same',
                                       kernel_initializer='he_normal',
                                       use_bias=False)(model_input)
        layer = tf.keras.layers.BatchNormalization()(layer)
        out_12 = tf.keras.layers.ReLU()(layer)

        layer = tf.keras.layers.Conv2D(256, kernel_size=3,
                                       dilation_rate=18, padding='same',
                                       kernel_initializer='he_normal',
                                       use_bias=False)(model_input)
        layer = tf.keras.layers.BatchNormalization()(layer)
        out_18 = tf.keras.layers.ReLU()(layer)

        layer = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1,
                                                      out_6, out_12,
                                                      out_18])

        layer = tf.keras.layers.Conv2D(256, kernel_size=1,
                                       dilation_rate=1, padding='same',
                                       kernel_initializer='he_normal',
                                       use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        model_output = tf.keras.layers.ReLU()(layer)
        return model_output

    def DeeplabV3Plus(nclasses=19):
        model_input = tf.keras.Input(shape=(img_h, img_w, 3))
        resnet50 = tf.keras.applications.ResNet50(weights='imagenet',
                                                  include_top=False,
                                                  input_tensor=model_input)
        layer = resnet50.get_layer('conv4_block6_2_relu').output
        layer = AtrousSpatialPyramidPooling(layer)
        input_a = tf.keras.layers.UpSampling2D(size=(img_h // 4 // layer.shape[1],
                                                     img_w // 4 // layer.shape[2]),
                                               interpolation='bilinear')(layer)

        input_b = resnet50.get_layer('conv2_block3_2_relu').output
        input_b = tf.keras.layers.Conv2D(48, kernel_size=(1, 1), padding='same',
                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                         use_bias=False)(input_b)
        input_b = tf.keras.layers.BatchNormalization()(input_b)
        input_b = tf.keras.layers.ReLU()(input_b)

        layer = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])

        layer = tf.keras.layers.Conv2D(256, kernel_size=3,
                                       padding='same', activation='relu',
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)
        layer = tf.keras.layers.Conv2D(256, kernel_size=3,
                                       padding='same', activation='relu',
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)
        layer = tf.keras.layers.UpSampling2D(size=(img_h // layer.shape[1],
                                                   img_w // layer.shape[2]),
                                             interpolation='bilinear')(layer)
        model_output = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1),
                                              padding='same')(layer)
        return tf.keras.Model(inputs=model_input, outputs=model_output)

    return DeeplabV3Plus(num_classes)


model = get_model()
model.summary()
# tf.keras.utils.plot_model(model, to_file="model.png",show_shapes=True, dpi=64)

# loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

loss = SparseCategoricalFocalLoss(from_logits=True, gamma=2)

step_per_epoch = len(train_images_folder)//batch_size
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
              loss=loss, metrics=['accuracy'])


def scheduler(epoch, lr):
    if epoch < 4:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

filepath = 'trained/deeplab_model1'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                monitor='loss',
                                                save_best_only=True,
                                                mode='min')

model.fit(train_dataset, steps_per_epoch=step_per_epoch, epochs=epochs,
          validation_data=valid_dataset,
          validation_steps=len(valid_images_folder)//batch_size,
          callbacks=[checkpoint, callback])

write_model_path= 'trained/deeplab_final'
model.save(write_model_path)
