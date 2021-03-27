import pandas as pd
import numpy as np
import cv2
import matplotlib as mlp
import matplotlib.pyplot as plt
import tensorflow as tf


class CarPrediction:

    # dimensions of our images.
    img_width, img_height = 64, 64
    channels = 3
    input_shape = (img_width, img_height, channels)


    def train(self):
        """
            This function train the CNN model taking from directory the images.
            It also makes data augmentation
        :return:
        """
        # parameters and constants
        train_data_dir = 'object_prediction/datasets/car-general/training'
        validation_data_dir = 'object_prediction/datasets/car-general/validation'
        epochs = 20
        batch_size = 64

        # getting images for training
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1 / 255,
            zoom_range=.2,
            height_shift_range=.2,
            width_shift_range=.2,
            horizontal_flip=True,
            fill_mode='nearest',
            shear_range=.2
        )

        training_data_gen = data_gen.flow_from_directory(
            train_data_dir,
            class_mode='binary',
            target_size=(self.img_width, self.img_height),
            batch_size=batch_size
        )

        testing_data_gen = data_gen.flow_from_directory(
            validation_data_dir,
            class_mode='binary',
            target_size=(self.img_width, self.img_height),
            batch_size=batch_size
        )

        # model tuning
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            training_data_gen,
            validation_data=testing_data_gen,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        output_values = history.history

        acc = output_values['accuracy']
        val_acc = output_values['val_accuracy']
        loss = output_values['loss']
        val_loss = output_values['val_loss']

        # print('accuracy = ' + acc + '\nvalidation accuracy = ' + val_acc + '\nloss = ' + loss + '\nvalidation loss = ' + val_loss)

        model.save('object_prediction/model_data/obj_recognition_model')
        model.save_weights('object_prediction/model_data/weights.h5')

    def predict(self, image_path):
        """
            Predict an image, if it is a car or not, giving the path to image
        """
        model_dir = 'object_prediction/model_data/obj_recognition_model'
        model = tf.keras.models.load_model(model_dir)

        # if needet - utils
        # print(model.summary())
        # plot_model(model, to_file='object_prediction/model_plot.png', show_shapes=True, show_layer_names=True, )

        # works
        # image = cv2.imread(prediction_dir)
        # image_array = np.asarray(image)

        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(self.img_width, self.img_height))

        # vectoring the image and standardizing it
        image_array = tf.keras.preprocessing.image.img_to_array(image)  # (height, width, channels)
        image_array /= 255.


        if self.show_image_to_predict:
            # imshow expects values in the range [0, 1]
            plt.imshow(image_array)
            plt.title('Image to predict')
            plt.show()

        # image_array = np.expand_dims(image_array, axis=0)                                                         # (1, height, width, channels)
        reshaped_image = image_array.reshape(1, self.img_height, self.img_width, self.channels)                     # (1, height, width, channels)
        prediction = model.predict(reshaped_image)
        return prediction[0][0] # ritorna solo il valore interessato

    def __init__(self, show_image_to_predict):
        """
        :param show_image_to_predict: if the program should show the image that we are going to predict
        :param show_true_image: if the program should show the image containing the image with label true or not (label true means a car)
        """
        self.show_image_to_predict = show_image_to_predict