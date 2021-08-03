import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras.utils.vis_utils import plot_model


"""

            ONGOING
            NOT FINISHED


"""


def preprocess_image(images):
    images = images.astype("float32") / 255
    # images = np.expand_dims(images, axis=1)
    return images


class ImoNet(tf.keras.Model):
    def __init__(self, filters, kernel_size, num_classes):
        super(ImoNet, self).__init__()
        self.block1 = NestedBlock(filters, kernel_size + 2)
        self.block2 = NestedBlock(filters, kernel_size)
        self.dense1 = tf.keras.layers.Dense(units=1024, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=512, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=512, activation='relu')
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', strides=(2, 2))
        self.do = tf.keras.layers.Dropout(0.2)
        self.mp = tf.keras.layers.MaxPool2D()
        self.flat = tf.keras.layers.Flatten()
        self.result = tf.keras.layers.Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.block1(x)
        x = self.block2(x)
        x = self.flat(x)
        x = self.result(x)
        return x


class NestedBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(NestedBlock, self).__init__(name='')
        self.conv_1x1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', name='Conv2D_1x1')
        self.conv_nxn = tf.keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), padding='same',
                                               name='Conv2D_nxn')
        self.conv_1xn = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, kernel_size), padding='same',
                                               name='Conv2D_1x{}'.format(kernel_size))
        self.conv_nx1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size, 1), padding='same',
                                               name='Conv2D_{}x1'.format(kernel_size))

        self.last_conv_1x1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), padding='same',
                                                    name='LastConv2D_1x1')
        self.last_conv_nxn = tf.keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size + 2, kernel_size + 2),
                                                    padding='same', name='LastConv2D_nxn')
        self.last_conv_1xn = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, kernel_size + 2), padding='same',
                                                    name='LastConv2D_1x{}'.format(kernel_size))
        self.last_conv_nx1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size + 2, 1), padding='same',
                                                    name='LastConv2D_{}x1'.format(kernel_size))

        self.mp = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.ap = tf.keras.layers.AvgPool2D()
        self.conc = tf.keras.layers.Concatenate()
        self.do = tf.keras.layers.Dropout(0.2)

    def call(self, inputs):
        first_branch = self.conv_1x1(inputs)
        first_branch = self.conv_nxn(first_branch)
        first_1_sub_branch = self.conv_1xn(first_branch)
        first_2_sub_branch = self.conv_nx1(first_branch)

        second_branch = self.conv_1x1(inputs)
        second_1_sub_branch = self.conv_1xn(second_branch)
        second_2_sub_branch = self.conv_nx1(second_branch)

        third_branch = self.mp(inputs)
        third_branch = self.conv_1x1(third_branch)

        fourth_branch = self.conv_1x1(inputs)

        conc = self.conc([first_1_sub_branch, first_2_sub_branch,
                          second_1_sub_branch, second_2_sub_branch,
                          third_branch, fourth_branch])

        final_path = self.mp(conc)
        final_path = self.last_conv_1x1(final_path)

        return conc


def run():
    (train_img, train_lbl), (val_img, val_lbl) = tf.keras.datasets.cifar10.load_data()

    train_img = preprocess_image(train_img)
    val_img = preprocess_image(val_img)

    imonet = ImoNet(10, 3, 10)

    # plot_model(imonet, show_shapes=True, show_layer_names=True, to_file='imonet.png', expand_nested=True)
    imonet.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    print("Starting fit the model")
    imonet.fit(x=train_img, y=train_lbl, validation_data=(val_img, val_lbl), epochs=10, verbose=1)
    print("Finished to fit the model")