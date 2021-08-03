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
    #images = np.expand_dims(images, axis=1)
    return images

def build_model(inputs):
    first_layer = tf.keras.layers.GlobalAveragePooling2D() (inputs)

    # branching
    first_branch = tf.keras.layers.Conv2D(filters=10, kernel_size=(1,1), padding='same') (first_layer)
    first_branch = tf.keras.layers.BatchNormalization() (first_branch)
    first_branch = tf.keras.layers.Conv2D(filters=10, kernel_size=(3,1), padding='same') (first_branch)
    first_branch = tf.keras.layers.Conv2D(filters=10, kernel_size=(1,3), padding='same') (first_branch)

    second_branch = tf.keras.layers.Conv2D(filters=10, kernel_size=(1,1), padding='same') (first_layer)
    second_branch = tf.keras.layers.BatchNormalization() (second_branch)
    second_branch = tf.keras.layers.Conv2D(filters=10, kernel_size=(1,3), padding='same') (second_branch)
    second_branch = tf.keras.layers.Conv2D(filters=10, kernel_size=(3,1), padding='same') (second_branch)
    second_branch = tf.keras.layers.Conv2D(filters=10, kernel_size=(1,3), padding='same') (second_branch)
    second_branch = tf.keras.layers.Conv2D(filters=10, kernel_size=(3,1), padding='same') (second_branch)

    x = tf.keras.layers.Flatten() ([first_branch, second_branch])
    x = tf.keras.layers.Dense(units=1024, activation='relu') (x)
    x = tf.keras.layers.Dense(units=512, activation='relu') (x)
    result = tf.keras.layers.Dense(units=10, activation='softmax') (x)

    model = tf.keras.models.Model(inputs=first_branch, output=result)
    return model



class ImoNet(tf.keras.Model):
    def __init__(self, filters, kernel_size, num_classes):
        super(ImoNet, self).__init__()
        self.block1 = NestedBlock(filters, kernel_size + 2)
        self.block2 = NestedBlock(filters, kernel_size)
        self.dense1 = tf.keras.layers.Dense(units=1024, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=512, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=512, activation='relu')
        self.do = tf.keras.layers.Dropout(0.2)
        self.mp = tf.keras.layers.MaxPool2D()
        self.flat = tf.keras.layers.Flatten()
        self.result = tf.keras.layers.Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        x = self.block1 (inputs)
        x = self.block2 (x)
        x = self.flat (x)
        x = self.dense1 (x)
        x = self.dense2 (x)
        # x = self.mp (x)
        x = self.result (x)
        return x



class NestedBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(NestedBlock, self).__init__(name='')
        self.conv_1x1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1,1), padding='same', name='Conv2D_1x1')
        self.conv_1xn = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, kernel_size), padding='same', name='Conv2D_1x{}'.format(kernel_size))
        self.conv_nx1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size, 1), padding='same', name='Conv2D_{}x1'.format(kernel_size))
        self.bn = tf.keras.layers.BatchNormalization()
        self.mp = tf.keras.layers.MaxPool2D()
        self.ap = tf.keras.layers.AvgPool2D()
        self.conc = tf.keras.layers.Add()
        self.do = tf.keras.layers.Dropout(0.2)

    def call(self, inputs):
        first_branch = self.conv_1x1 (inputs)
        first_branch = self.ap (first_branch)
        first_branch = self.conv_1xn (first_branch)
        first_branch = self.conv_nx1 (first_branch)
        first_branch = self.ap (first_branch)

        second_branch = self.conv_1x1 (inputs)
        second_branch = self.ap (second_branch)
        second_branch = self.conv_1xn (second_branch)
        second_branch = self.conv_nx1 (second_branch)
        second_branch = self.ap (second_branch)

        conc = self.conc([first_branch, second_branch])
        return conc









def run():
    (train_img, train_lbl) , (val_img, val_lbl) = tf.keras.datasets.cifar10.load_data()

    train_img = preprocess_image(train_img)
    val_img = preprocess_image(val_img)

    imonet = ImoNet(10, 3, 10)
    # imonet = build_model(train_img)
    plot_model(imonet, show_shapes=True, show_layer_names=True, to_file='imonet.png', expand_nested=True)
    imonet.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    print("x")
    imonet.fit(x=train_img, y=train_lbl, validation_data=(val_img, val_lbl), epochs=2, verbose=1)
    print("xx")














