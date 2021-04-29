import numpy as np
import tensorflow as tf
from glob import glob
import os
from focal_loss import SparseCategoricalFocalLoss

img_w, img_h = 704, 704
batch_size = 4
num_classes = 19
# DATA_DIR = "/home/minouei/Downloads/datasets/contract/version5"
DATA_DIR = "/netscratch/minouei/versicherung/version5"
epochs = 4

slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=15000)
strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)
strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.get_strategy()
task_type, task_id = slurm_resolver.get_task_info()

def _is_chief(task_type, task_id):
    return task_type is None or task_type == 'chief' or (task_type == 'worker' and
                                                         task_id == 0)


def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir


def write_filepath(filepath, task_type, task_id):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_chief(task_type, task_id):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)




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


@tf.function
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


def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# average across the global batch and reduce down to a scalar
# def compute_loss(labels, predictions):
#         per_example_loss = loss_object(labels, predictions)
#         return tf.nn.compute_average_loss(
#             per_example_loss, global_batch_size=global_batch_size)

@tf.function
def train_step(inputs):
    inputs, labels = inputs

    with tf.GradientTape() as tape:
        y_pred = checkpoint.model(inputs, training=True)
        loss_value = loss_object(labels, y_pred)# [NxHxWx1]
        # average across the batch (N) with the approprite global batch size
        loss_value = tf.reduce_sum(loss_value, axis=0) / global_batch_size
        # reduce down to a scalar (reduce H, W)
        loss_value = tf.reduce_mean(loss_value)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, checkpoint.model.trainable_variables)
    # grads, _ = tf.clip_by_global_norm(grads, clip_norm=tf.constant(10.0)) #TODO
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    checkpoint.optimizer.apply_gradients(zip(grads, checkpoint.model.trainable_variables))
    #update metrics
    train_loss_metric.update_state(loss_value)
    train_acc_metric.update_state(labels, y_pred)

    return loss_value


@tf.function
def distributed_train_step(inputs):
    per_gpu_loss = strategy.run(train_step, args=(inputs,))
    loss_value = strategy.reduce(tf.distribute.ReduceOp.SUM, per_gpu_loss, axis=None)
    return loss_value


@tf.function
def test_step(inputs):
    inputs, labels = inputs
    y_pred = checkpoint.model(inputs, training=False)

    loss_value = loss_object(labels, y_pred)
    # average across the batch (N) with the approprite global batch size
    loss_value = tf.reduce_sum(loss_value, axis=0) / global_batch_size
    # reduce down to a scalar (reduce H, W)
    loss_value = tf.reduce_mean(loss_value)

    test_loss_metric.update_state(loss_value)
    test_acc_metric.update_state(labels, y_pred)

    return loss_value


@tf.function
def distributed_test_step(dist_strategy, inputs):
    per_gpu_loss = dist_strategy.run(test_step, args=(inputs,))
    loss_value = dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_gpu_loss, axis=None)
    return loss_value


def save_trained_model(write_model_path):
    checkpoint.model.save(write_model_path)
    if not is_chief:
        tf.io.gfile.rmtree(os.path.dirname(write_model_path))
    tf.__internal__.distribute.multi_process_runner.get_barrier().wait()


def save_checkpoint(write_model_path):
    checkpoint.model.save(write_model_path)
    if not is_chief:
        tf.io.gfile.rmtree(os.path.dirname(write_model_path))
    tf.__internal__.distribute.multi_process_runner.get_barrier().wait()


def restore_checkpoint():
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print("Model restored at epoch {}".format(checkpoint.epoch.numpy()))


with strategy.scope():
    model = get_model()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

    train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
    test_loss_metric = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_acc_metric = tf.keras.metrics.CategoricalAccuracy('test_accuracy')

    loss_object = SparseCategoricalFocalLoss(from_logits=True, gamma=2, reduction=tf.keras.losses.Reduction.NONE)
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    #     from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    # task_type = strategy.cluster_resolver.task_type
    # task_id = strategy.cluster_resolver.task_id

    print('task_type:{} task_id:{} num_replicas_in_sync:{}'.format(task_type, task_id,strategy.num_replicas_in_sync))
    # task_type, task_id = 'chief', 0
    is_chief = _is_chief(task_type, task_id)

    checkpoint = tf.train.Checkpoint(epoch=tf.Variable(1),
                             val_loss=tf.Variable(np.inf), model=model, optimizer=optimizer)
    write_model_path = write_filepath('slurm_saved_models', task_type, task_id)
    write_checkpoint_dir = write_filepath('slurm_checkpoints', task_type, task_id)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                directory=write_checkpoint_dir,
                                                max_to_keep=3)

    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    valid_dataset = strategy.experimental_distribute_dataset(valid_dataset)
    
    

    # if config.trainer.epoch_to_continue > 0:
    #     with strategy.scope():
    #         restore_checkpoint()

    for epoch in range(epochs):
        for step, inputs in enumerate(train_dataset):
            loss = distributed_train_step(inputs)

        for step, inputs in enumerate(valid_dataset):
            loss = distributed_test_step(inputs)


        template = ('Epoch: {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}')
        print(template.format(epoch, train_loss_metric.result(),
                          train_acc_metric.result(),
                          test_loss_metric.result(),
                          test_acc_metric.result()))


        train_loss_metric.reset_states()
        train_acc_metric.reset_states()
        test_loss_metric.reset_states()
        test_acc_metric.reset_states()

        # save_checkpoint()

    save_trained_model(write_model_path)
