import object_prediction.core as obj
import tkinter as tk
import tkinter.filedialog as fd


if __name__ == '__main__':
    car_prediction = obj.CarPrediction(show_image_to_predict=True)
    # car_prediction.train()

    # get the image/s
    root = tk.Tk()
    root.withdraw()
    paths = fd.askopenfilenames(parent=root, title='Choose a file')

    # predict
    for p in paths:
        print("The image", "is" if car_prediction.predict(p) > 0.6 else "is not", "a car")