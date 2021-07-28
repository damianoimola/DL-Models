import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydot
from tensorflow.python.keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split



# utils
def format_output(data):
    y1 = data.pop('Y1')
    y1 = np.array(y1)
    y2 = data.pop('Y2')
    y2 = np.array(y2)
    return y1, y2

def normalize(data, stats):
    return (data - stats['mean']) / stats['std']

def plot_diff(y_true, y_pred, title=''):
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.plot([-100, 100], [-100, 100])
    plt.show()


# model with functional API
def build_model(numbers_of_labels):
    # creating a model with functional API
    input_layer = tf.keras.Input(shape=(numbers_of_labels,))
    first_dense = tf.keras.layers.Dense(units='128', activation=tf.nn.relu) (input_layer)
    second_dense = tf.keras.layers.Dense(units='128', activation=tf.nn.relu) (first_dense)

    # branching
    first_output = tf.keras.layers.Dense(units='1', name='first_output') (second_dense)
    third_dense = tf.keras.layers.Dense(units='64', activation=tf.nn.relu) (second_dense)

    second_output = tf.keras.layers.Dense(units='1', name='second_output') (third_dense)

    # compose the model
    func_model = tf.keras.models.Model(inputs=input_layer, outputs=[first_output, second_output])

    print(func_model.summary())

    return func_model



def run():
    # loading datafrme
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
    data = pd.read_excel(URL)

    # splitting into train and test data
    train, test = train_test_split(data, test_size=0.2)
    train_stats = train.describe()
    train_stats.pop('Y1')
    train_stats.pop('Y2')
    train_stats = train_stats.transpose()

    # getting labels
    train_labels = format_output(train)
    test_labels = format_output(test)

    # normalize
    norm_train = normalize(train, train_stats)
    norm_test = normalize(test, train_stats)

    model = build_model(len(train.columns))

    # plotting model arch
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='simple_classifier/model.png')

    # define the model works
    optimizer = tf.keras.optimizers.SGD(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss={'first_output' : 'mse',
                        'second_output' : 'mse'},
                  metrics={
                      'first_output': tf.keras.metrics.RootMeanSquaredError(),
                      'second_output': tf.keras.metrics.RootMeanSquaredError(),
                  })

    # train the model
    history = model.fit(norm_train, train_labels, epochs=2000, batch_size=10, validation_data=(norm_test, test_labels))
    loss, first_loss, second_loss, first_rmse, second_rmse = model.evaluate(x=norm_test, y=test_labels)

    # predict
    pred = model.predict(norm_test)
    plot_diff(test_labels[0], pred[0], title='Y1')
    plot_diff(test_labels[1], pred[1], title='Y2')