import tensorflow as tf
import os.path
from load_data import Data
from create_model import Model
from knockknock import discord_sender
import numpy as np

tf.config.run_functions_eagerly(True)

webhook_url = 'https://discord.com/api/webhooks/1249032522969645116/j0HgZv0za49WZ-LlypJts--DRDszZcNaKJyJrbntnxz4wvppVXFHk7A49MLBAu7nndQg'


@discord_sender(webhook_url=webhook_url)
def main_model(
    train=False,
    evaluate=False,
    predict=False,
    specific_set='',
    epoch_try=1000,
    safe_data_load=True,
    show_activation=False,
    fold=3,
    heat_map=True,
):
    """
    :param train: whether to train or not, will use the data inside dataset/specific_set folder
    :param evaluate: whether to evaluate or not, will use the data inside dataset/specific_set folder
    :param predict: whether to predict or not, it will predict for each images of the dataset/tester folder
    :param specific_set: name of the sub-folder (inside /dataset) containing the data
    :param epoch_try: number of epochs of the fold method
    :param safe_data_load: whether to check the data before loading it
    :param show_activation: whether to show the activation evolution or not
    """
    print(f'Starting {specific_set}')
    try:
        input_shape = (224, 224, 3)
        batch_size = 30

        data = Data(
            os.path.join('dataset', specific_set),
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
                    specific_set,
                    f'{len(data.get_classes().keys())}_classes_model.h5',
                )
            )

        if show_activation:
            model.show_activation_evolution()

        if train:
            if fold > 1:
                print('Searching best number of epochs...')
                best_epoch = model.get_best_epoch(data, k=fold, epochs=epoch_try)
                if best_epoch == -1:
                    return -1
            else:
                best_epoch = epoch_try

            if not os.path.exists(os.path.join('models')):
                os.mkdir(os.path.join('models'))

            if not os.path.exists(os.path.join('models', specific_set)):
                os.mkdir(os.path.join('models', specific_set))

            model.plot_folds(
                save_file_name=os.path.join(
                    'models', specific_set, f'{fold}_fold_val_loss_plot.png'
                )
            )
            print(f'Best number of epochs: {best_epoch}')
            data.reload_generator()

            print('Starting the training...')
            model.train(data=data, epochs=best_epoch, batch_size=batch_size)

            model.save(
                path=os.path.join(
                    'models',
                    specific_set,
                    f'{len(data.get_classes().keys())}_classes_model.h5',
                )
            )

            model.plot_training(
                'loss',
                save_file_name=os.path.join('models', specific_set, 'loss_plot.png'),
            )
            model.plot_training(
                'accuracy',
                save_file_name=os.path.join(
                    'models', specific_set, 'accuracy_plot.png'
                ),
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
                pred = model.predict(file_path, heat_map=heat_map)
                index = np.where(pred[0] == max(pred[0]))
                first_pred = pred[0][index[0][0]]
                pred[0][index[0][0]] = -1
                index2 = np.where(pred[0] == max(pred[0]))
                second_pred = pred[0][index2[0][0]]
                for key1, i in data.get_classes().items():
                    if i == index[0][0]:
                        for key2, i in data.get_classes().items():
                            if i == index2[0][0]:
                                print(
                                    f'{file_path} predicted as {key1} with {first_pred: .4f} precision then as {key2} with {second_pred: .4f}'
                                )
                                break
        return 0
    except Exception as e:
        print(e.with_traceback())
        return -1


if __name__ == '__main__':
    main_model(
        train=True,
        evaluate=True,
        predict=True,
        specific_set='clothes',
        epoch_try=100,
        safe_data_load=False,
        show_activation=True,
        fold=3,
        heat_map=True,
    )
    # main_model(train=False, evaluate=False, predict=False, specific_set='shape', epoch_try=50, safe_data_load=False, show_activation=True)
    # main_model(train=True, evaluate=True, predict=False, specific_set='symboles', epoch_try=1000, safe_data_load=False, show_activation=True, fold=True)
    # main_model(train=False, evaluate=True, predict=True, specific_set='color', epoch_try=750, safe_data_load=False, show_activation=False, fold=2, heat_map=False)
