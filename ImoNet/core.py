import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

"""

            ONGOING
            NOT FINISHED


"""

# usefull methods

def preprocess_image(images):
    images = images.astype("float32") / 255
    # images = np.expand_dims(images, axis=1)
    return images

def plot_data(data):
  plt.plot(data['loss'], label="loss")
  plt.plot(data['accuracy'], label="accuracy")
  plt.plot(data['val_loss'], label="val_loss")
  plt.plot(data['val_accuracy'], label="val_accuracy")
  #plt.plot(data['loss'], data['accuracy'], data['val_loss'], data['val_accuracy'])
  plt.legend()






"""
    Bad Acc and Loss
"""
class ImoNet_Test(tf.keras.Model):
    def __init__(self, filters, kernel_size, num_classes):
        super(ImoNet_Test, self).__init__()
        self.block1 = NestedBlock(filters, kernel_size + 2)
        self.block2 = NestedBlock(filters * 2, kernel_size)
        self.dense1 = tf.keras.layers.Dense(units=1024, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=512, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=512, activation='relu')
        self.conv = tf.keras.layers.Conv2D(filters=filters * 2, kernel_size=kernel_size, padding='same', strides=(2, 2))
        self.do = tf.keras.layers.Dropout(0.2)
        self.mp = tf.keras.layers.MaxPool2D()
        self.flat = tf.keras.layers.Flatten()
        self.result = tf.keras.layers.Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.block1(x)
        x = self.block2(x)
        x = self.flat(x)
        x = self.dense2(x)
        x = self.mp(x)
        x = self.result(x)
        return x

class NestedBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(NestedBlock, self).__init__()
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



"""
    TRYING (Full conv, no dense layers. Conv 1x1 as labeler)
"""
class W_Net(tf.keras.Model):
    def __init__(self, filters, kernel_size, num_classes):
        super(W_Net, self).__init__()
        # utils
        self.conc = tf.keras.layers.Concatenate()
        self.pred = tf.keras.layers.Conv2D(kernel_size=1, filters=filters)
        # encoders
        self.fe1_1 = FeatureEncoder(filters, kernel_size)
        self.fe1_2 = FeatureEncoder(filters * 2, kernel_size)
        self.fe1_3 = FeatureEncoder(filters * 4, kernel_size)
        self.fe1_4 = FeatureEncoder(filters * 6, kernel_size)
        # bottleneck
        self.btlneck1 = tf.keras.layers.Conv2D(filters=filters * 8, kernel_size=kernel_size, padding='same',
                                               name='Conv2D_btlneck1_{}x{}'.format(kernel_size, kernel_size))
        # decoders
        self.fd1_4 = FeatureDecoder(filters, kernel_size * 6)
        self.fd1_3 = FeatureDecoder(filters, kernel_size * 4)
        # bottleneck
        self.btlneck2 = tf.keras.layers.Conv2D(filters=filters * 2, kernel_size=kernel_size, padding='same',
                                               name='Conv2D_btlneck2_{}x{}'.format(kernel_size, kernel_size))
        # encoders
        self.fe2_3 = FeatureEncoder(filters * 4, kernel_size)
        self.fe2_4 = FeatureEncoder(filters * 6, kernel_size)
        # bottleneck
        self.btlneck3 = tf.keras.layers.Conv2D(filters=filters * 28, kernel_size=kernel_size, padding='same',
                                               name='Conv2D_btlneck3_{}x{}'.format(kernel_size, kernel_size))
        # decoders
        self.fd2_4 = FeatureDecoder(filters, kernel_size * 6)
        self.fd2_3 = FeatureDecoder(filters, kernel_size * 4)
        self.fd2_2 = FeatureDecoder(filters, kernel_size * 2)
        self.fd2_1 = FeatureDecoder(filters, kernel_size)

    def call(self, inputs):
        fe1_1 = self.fe1_1(inputs)
        fe1_2 = self.fe1_2(fe1_1)
        fe1_3 = self.fe1_3(fe1_2)
        fe1_4 = self.fe1_4(fe1_3)
        btlneck1 = self.btlneck1(fe1_4)

        conc1 = self.conc([fe1_4, btlneck1])

        fd1_4 = self.fd1_4(conc1)
        fd1_3 = self.fd1_4(fd1_4)
        btlneck2 = self.btlneck2(fe1_3)

        conc2 = self.conc([fe1_3, btlneck2])

        fe2_3 = self.fe2_3(conc2)
        fe2_4 = self.fe2_4(fe2_3)
        btlneck3 = self.btlneck3(fe2_4)

        conc3 = self.conc([fe2_4, btlneck3])

        fd2_4 = self.fd2_4(conc3)
        fd2_3 = self.fd2_3(fd2_4)
        fd2_2 = self.fd2_2(fd2_3)
        fd2_1 = self.fd2_1(fd2_2)

        pred = self.pred(fd2_1)
        return pred

class FeatureEncoder(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(FeatureEncoder, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                            name='Conv2D_1_{}x{}'.format(kernel_size, kernel_size))
        self.conv2 = tf.keras.layers.Conv2D(filters=filters * 2, kernel_size=kernel_size, padding='same',
                                            name='Conv2D_2_{}x{}'.format(kernel_size, kernel_size))
        self.act = tf.keras.layers.Activation('relu')
        self.avgp = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
        self.do = tf.keras.layers.Dropout(0.2)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.act(x)
        x = self.avgp(x)
        return x

class FeatureDecoder(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(FeatureDecoder, self).__init__()
        self.convt1 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding='same',
                                                      name='Conv2DT_1_{}x{}'.format(kernel_size, kernel_size))
        self.convt2 = tf.keras.layers.Conv2DTranspose(filters=filters / 2, kernel_size=kernel_size, padding='same',
                                                      name='Conv2DT_2_{}x{}'.format(kernel_size, kernel_size))
        self.act = tf.keras.layers.Activation('relu')
        self.avgp = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
        self.do = tf.keras.layers.Dropout(0.2)

    def call(self, inputs):
        x = self.convt1(inputs)
        x = self.convt2(x)
        x = self.act(x)
        x = self.avgp(x)
        return x



"""
    VGG CLONE APPROX 40 EPOCHS
    Test with increasing DropOut value (cap @ 0.4)
    
    NOW: Acc: 0.89; Loss: 0.43
"""
class ImoNet(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(ImoNet, self).__init__()
        self.conv1_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                              name='Conv2D_1_1_{}x{}'.format(kernel_size, kernel_size),
                                              input_shape=(32, 32, 3))
        self.conv1_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                              name='Conv2D_1_2_{}x{}'.format(kernel_size, kernel_size))
        self.conv2_1 = tf.keras.layers.Conv2D(filters=filters * 2, kernel_size=kernel_size, padding='same',
                                              name='Conv2D_2_1_{}x{}'.format(kernel_size, kernel_size))
        self.conv2_2 = tf.keras.layers.Conv2D(filters=filters * 2, kernel_size=kernel_size, padding='same',
                                              name='Conv2D_2_2_{}x{}'.format(kernel_size, kernel_size))
        self.conv3_1 = tf.keras.layers.Conv2D(filters=filters * 4, kernel_size=kernel_size, padding='same',
                                              name='Conv2D_3_1_{}x{}'.format(kernel_size, kernel_size))
        self.conv3_2 = tf.keras.layers.Conv2D(filters=filters * 4, kernel_size=kernel_size, padding='same',
                                              name='Conv2D_3_2_{}x{}'.format(kernel_size, kernel_size))

        self.mp1 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.mp2 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.mp3 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.do2 = tf.keras.layers.Dropout(0.2)
        self.do3 = tf.keras.layers.Dropout(0.2)
        self.do4 = tf.keras.layers.Dropout(0.2)
        self.f = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu',
                                            activity_regularizer=tf.keras.regularizers.L2(0.01))
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.mp1(x)
        x = self.do2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        # x = self.mp2(x)
        x = self.do3(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.mp3(x)
        x = self.do4(x)
        x = self.f(x)
        x = self.dense1(x)
        x = self.do4(x)
        x = self.dense2(x)
        return x




def run():
    (train_img, train_lbl), (val_img, val_lbl) = tf.keras.datasets.cifar10.load_data()

    train_img = preprocess_image(train_img)
    val_img = preprocess_image(val_img)

    #imonet = ImoNet_Test(256, 3, 10)
    imonet = ImoNet(32, 3)

    # plot_model(imonet, show_shapes=True, show_layer_names=True, to_file='imonet.png', expand_nested=True)
    imonet.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
                   loss=tf.keras.losses.sparse_categorical_crossentropy,
                   metrics=['accuracy'])

    print("Starting fit the model")
    history = imonet.fit(x=train_img, y=train_lbl, validation_data=(val_img, val_lbl), epochs=40, verbose=1)
    plot_data(history.history)
    print("Finished to fit the model")
