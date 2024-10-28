from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os.path
from load_data import Data
from create_model import Model
import numpy as np

tf.config.run_functions_eagerly(True)


def test_model(
    train=False,
    evaluate=False,
    predict=False,
    model_name='',
    epoch_try=1000,
    safe_data_load=True,
    show_activation=False,
) -> int:
    """
    allow to use a model for training, prediction or evaluation
    :param train: whether to train or not, will use the data inside dataset/model_name folder
    :param evaluate: whether to evaluate or not, will use the data inside dataset/model_name folder
    :param predict: whether to predict or not, it will predict for each images of the dataset/tester folder
    :param model_name: name of the sub-folder (inside /dataset) containing the data, it will give its name to the model
    :param epoch_try: number of epochs of the fold method
    :param safe_data_load: whether to check the data before loading it
    :param show_activation: whether to show the activation evolution or not
    :return 0 is everything is ok, -1 if an error appended
    """
    print(f'Starting {model_name}')
    try:
        input_shape = (224, 224, 3)
        batch_size = 10

        data = Data(
            os.path.join('dataset', model_name),
            input_shape=input_shape,
            batch_size=batch_size,
            safe=safe_data_load,
        )

        if train:
            model = Model(
                input_shape=data.train_generator.image_shape,
                nb_classes=len(data.get_classes().keys()),
            )
        else:
            model = Model(
                path=os.path.join(
                    'models',
                    model_name,
                    f'{len(data.get_classes().keys())}_classes_model.h5',
                )
            )

        if show_activation:
            model.show_activation_evolution()

        if train:
            print('Searching best number of epochs...')
            k = 3
            best_epoch = model.get_best_epoch(data, k=k, epochs=epoch_try)
            if best_epoch == -1:
                return -1

            if not os.path.exists(os.path.join('models')):
                os.mkdir(os.path.join('models'))

            if not os.path.exists(os.path.join('models', model_name)):
                os.mkdir(os.path.join('models', model_name))

            model.plot_folds(
                save_file_name=os.path.join(
                    'models', model_name, f'{k}_fold_val_loss_plot.png'
                )
            )
            print(f'Best number of epochs: {best_epoch}')
            data.reload_generator()

            print('Starting the training...')
            model.train(data=data, epochs=best_epoch, batch_size=batch_size)

            model.save(
                path=os.path.join(
                    'models',
                    model_name,
                    f'{len(data.get_classes().keys())}_classes_model.h5',
                )
            )

            model.plot_training(
                'loss',
                save_file_name=os.path.join('models', model_name, 'loss_plot.png'),
            )
            model.plot_training(
                'accuracy',
                save_file_name=os.path.join('models', model_name, 'accuracy_plot.png'),
            )

            if evaluate:
                data.reload_generator()

        if evaluate:
            model.evaluate(data=data, batch_size=batch_size)

        if predict:
            tf.data.experimental.enable_debug_mode()

            test_image = os.listdir(os.path.join('dataset', 'tester'))
            for image_name in test_image:
                file_path = os.path.join('dataset', 'tester', image_name)
                pred = model.predict(
                    image.load_img(
                        file_path, target_size=(input_shape[0], input_shape[1])
                    ),
                    heat_map=True,
                )
                index = np.where(pred[0] == max(pred[0]))
                for key, i in data.get_classes().items():
                    if i == index[0][0]:
                        print(
                            f'{file_path} predicted as {key} with {pred[0][i]: .4f} precision'
                        )
                        break
        return 0
    except Exception as e:
        print(e.with_traceback())
        return -1


if __name__ == '__main__':
    test_model(
        train=False,
        evaluate=False,
        predict=True,
        model_name='clothes',
        epoch_try=500,
        safe_data_load=False,
        show_activation=True,
    )
    test_model(
        train=False,
        evaluate=False,
        predict=False,
        model_name='shape',
        epoch_try=50,
        safe_data_load=False,
        show_activation=True,
    )
