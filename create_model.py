import os.path
import sys

import numpy as np

from load_data import Data
from tensorflow.keras import models, callbacks
import matplotlib.pyplot as plt


class CustomCallback(callbacks.Callback):
    """Class which inherite from the Callback class, allows to create a custom callback function"""

    def __init__(self, max_epoch):
        """
        Constructor method that initializes the progress bar and max number of epochs
        :param max_epoch: number of iteration that will be done
        """
        super().__init__()
        self.max_epoch = max_epoch
        self.pbar = None

    def __del__(self):
        """
        Deconstructor to close the progress bar
        """
        if self.pbar is not None:
            self.pbar.close()

    def on_epoch_end(self, epoch, logs=None):
        """
        Method ran when an epoch is ending, this method make the progress bar progress
        :param epoch: actual epoch number
        :param logs: ...
        """
        from tqdm import tqdm

        if self.pbar is None:
            self.pbar = tqdm(total=self.max_epoch, file=sys.stdout)
        self.pbar.update(n=1)


def get_img_tensor(img):
    from tensorflow.keras.preprocessing import image

    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255
    return img_tensor


def make_gradcam_heatmap(img_array, model, last_conv_layer, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = models.Model(model.inputs, [last_conv_layer.output, model.output])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    from tensorflow import (
        GradientTape,
        argmax,
        reduce_mean,
        squeeze,
        math,
        maximum,
        newaxis,
    )

    with GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., newaxis]
    heatmap = squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = maximum(heatmap, 0) / math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path='cam.jpg', alpha=0.4):
    from tensorflow.keras.preprocessing import image

    img = image.load_img(img_path)
    img = image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    import matplotlib as mpl

    jet = mpl.colormaps['jet']

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = image.array_to_img(superimposed_img)
    fig, ax = plt.subplots()
    ax.imshow(superimposed_img)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


class Model:
    """Class to create, train or visualize a CNN model"""

    def __init__(self, input_shape=None, nb_classes=None, path=None):
        """
        :param input_shape: input shape of the model, only needed if path is not provided
        :param nb_classes: number of classes handled by the model (ex: t-shirt, pants, sweatshirt = 3 classes), only needed if path is not provided
        :param path: path to the model to load
        """
        self.train_history = None
        self.all_loss_history = None
        self.input_shape = input_shape
        self.model = None
        self.model_name = os.path.basename(path)
        if path is None and nb_classes is not None and input_shape is not None:
            self._build_model(nb_classes)
        else:
            self._load_model(path)

    def _load_model(self, path):
        """
        This method loads a model from a .h5 file
        It is run from the constructor if a path was given
        :param path: path to the model to load
        """
        self.model = models.load_model(path)
        self.input_shape = self.model.layers[0].input.shape[1:]

    def _build_model(self, nb_classes):
        """
        This method build a CNN model using keras
        It is run from the constructor if a path was not given
        The model use the following pattern: Conv Conv MaxPool Conv Conv MaxPool Conv Conv MaxPool
        :param nb_classes: number of classes used by the final softmax activation function
        """
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
        from tensorflow.keras import optimizers

        self.model = models.Sequential(
            [
                Conv2D(
                    32,
                    kernel_size=(3, 3),
                    activation='relu',
                    input_shape=self.input_shape,
                ),
                Conv2D(32, kernel_size=(3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2), strides=2),
                Conv2D(64, kernel_size=(3, 3), activation='relu'),
                Conv2D(64, kernel_size=(3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2), strides=2),
                Conv2D(128, kernel_size=(3, 3), activation='relu'),
                Conv2D(128, kernel_size=(3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2), strides=2),
                Flatten(),
                Dense(512, activation='relu'),
                Dense(nb_classes, activation='softmax'),
            ]
        )
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

    def train(self, data: Data, epochs, batch_size):
        """
        Train the model and store the last train history
        :param data: data that will be used for training and validation, it has to be of type Data
        :param epochs: number of epochs to train the model
        :param batch_size: batch size used to train the model
        """
        self.train_history = self.model.fit(
            data.train_generator,
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=data.train_set_length // epochs // batch_size,
            validation_data=data.validation_generator,
            validation_steps=data.validation_set_length // epochs // batch_size,
            verbose=2,
            callbacks=[callbacks.Callback(), callbacks.EarlyStopping(patience=50)],
        )
        return self.train_history

    def plot_folds(self, show=True, save_file_name=''):
        """
        Plot the last folds, return None if no training history is found
        :param show: whether to show the plot or not
        :param save_file_name: file name to save the plot, doesn't save if not specified
        """
        if self.all_loss_history is None:
            return None
        for history in self.all_loss_history:
            plt.plot(history)
        plt.title('val loss over epoch')
        plt.ylabel('Val Loss')
        plt.xlabel('Epoch')
        plt.legend([f'fold {i}' for i in range(len(self.all_loss_history))], loc=0)
        if save_file_name != '' and save_file_name.endswith('.png'):
            plt.savefig(save_file_name, format='png')
        if show:
            plt.show()
        plt.clf()

    def plot_training(self, metric, show=True, save_file_name=''):
        """
        Plot the last training, return None if no training history is found
        :param metric: metric to plot, can be 'accuracy' or 'loss'
        :param show: whether to show the plot or not
        :param save_file_name: file name to save the plot, doesn't save if not specified
        """
        if self.train_history is None:
            return None
        plt.plot(self.train_history.history[f'{metric}'])
        plt.plot(self.train_history.history[f'val_{metric}'])
        plt.title(metric)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc=0)
        if save_file_name != '' and save_file_name.endswith('.png'):
            plt.savefig(save_file_name, format='png')
        if show:
            plt.show()
        plt.clf()

    def predict(self, file_path, heat_map=False, verbose=2):
        """
        Predict the label of an image using actual model
        :file_path: file path of the image to predict
        :heat_map: whether to show the heat map or not
        :verbose: amount of information from the prediction to display (0: nothing, 1: everything, 2: important only)
        """
        from tensorflow.data.experimental import enable_debug_mode

        enable_debug_mode()

        img_tensor = get_img_tensor(file_path)
        prediction = self.model.predict(img_tensor, verbose=verbose)
        if heat_map:
            self._show_heatmap(file_path, prediction)
        return prediction

    def _show_heatmap(self, file_path, prediction):
        """
        Display the heatmap of the prediction of an image
        """
        img_array = get_img_tensor(file_path)
        last_conv_layer_name = self.model.layers[-5]
        heatmap = make_gradcam_heatmap(img_array, self.model, last_conv_layer_name)
        save_and_display_gradcam(file_path, heatmap)

    def get_best_epoch(self, data: Data, k, epochs):
        """
        Gets the best number of epochs by running k times the training
        :param data: data used for the training
        :param k: number of folds to split the data into
        :param epochs: number of epochs to execute for each folds
        """
        self.all_loss_history = []
        if (
            min(
                [
                    length // epochs // k
                    for length in [data.train_set_length, data.validation_set_length]
                ]
            )
            <= 1
        ):
            print(
                f'Too many epochs, max epoch for k = {k} -> {min([length // (k * 2) for length in [data.train_set_length, data.validation_set_length]])}'
            )
            return -1
        for i in range(k):
            print(f'Fold no. {i + 1} out of {k}')
            history = self.model.fit(
                data.train_generator,
                epochs=epochs,
                steps_per_epoch=data.train_set_length // epochs // k,
                validation_data=data.validation_generator,
                validation_steps=data.validation_set_length // epochs // k,
                verbose=0,
                callbacks=[CustomCallback(epochs)],
            )
            self.all_loss_history.append(history.history['val_loss'])
        min_epoch_number = min([len(x) for x in self.all_loss_history])
        average_loss_history = [
            np.mean([x[i] for x in self.all_loss_history])
            for i in range(min_epoch_number)
        ]
        best_epochs = np.argmin(average_loss_history) + 1
        return best_epochs

    def evaluate(self, data: Data, batch_size):
        """
        Evaluate the model
        :param data: data used to evaluate
        :param batch_size: batch size of the data evaluated
        """
        train_evaluation = self.model.evaluate(
            data.train_generator,
            steps=data.train_set_length // batch_size,
            batch_size=batch_size,
        )
        test_evaluation = self.model.evaluate(
            data.test_generator,
            steps=data.test_set_length // batch_size,
            batch_size=batch_size,
        )
        print(
            f'Train loss: {train_evaluation[0]}, train accuracy: {train_evaluation[1]}'
        )
        print(f'Test loss: {test_evaluation[0]}, test accuracy: {test_evaluation[1]}')
        return train_evaluation, test_evaluation

    def summarize(self):
        """Summarize the model layers and parameters"""
        self.model.summary()

    def show_activation_evolution(self):
        """Show the evolution of the activation for each layer of the model"""
        table = []
        for layer in self.model.layers:
            activation_size = 1
            for output_shape in layer.output_shape[1:]:
                activation_size *= output_shape
            table.append([layer.name, layer.output_shape, activation_size])
        from tabulate import tabulate

        print(tabulate(table, headers=['Name', 'Shape', 'Activation Size']))

    def save(self, path):
        """
        Save the model
        :param path: path to save the model
        """
        if path.endswith('.h5'):
            self.model.save(path)

    def get_input_shape(self):
        """
        Get the input shape
        :return: input shape of the model
        """
        return self.input_shape

    def get_model_name(self):
        """
        Get model name
        :return: name of the model
        """
        return self.model_name
