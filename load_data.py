import os
import sys


class Data:
    """Class to get the train, test and validation data from a directory (it can be get from a zip)"""

    def __init__(self, path, input_shape, batch_size, safe=True):
        """
        :param path: Path of the directory. The directory must contain 3 directories: train, test and validation. Each one of them must have a directory for each class containing the image of the class
        :param input_shape: shape to format the images to
        :param batch_size: batch size of the generators
        :param safe: whether to run a little program before to make sure that all images can be formatted correctly
        """
        import os

        self.dir_path = path

        if not os.path.isdir(self.dir_path):
            raise NotADirectoryError(f'{self.dir_path} is not a directory')
        if not os.path.isdir(os.path.join(self.dir_path, 'train')) or not os.path.isdir(
                os.path.join(self.dir_path, 'test')) or not os.path.isdir(os.path.join(self.dir_path, 'validation')):
            raise Exception('Bad directory content')

        self._train_dir = os.path.join(self.dir_path, 'train')
        self._test_dir = os.path.join(self.dir_path, 'test')
        self._validation_dir = os.path.join(self.dir_path, 'validation')

        self.input_shape = input_shape
        self.batch_size = batch_size

        if safe:
            if not self.check_data_format():
                raise Exception('Some files are not supported by this dataset')

        self._load_generator()

    def check_data_format(self):
        """Check whether the files contained by the directories will be usable"""
        import os
        from tqdm import tqdm
        from tensorflow.keras.preprocessing import image
        import itertools
        all_files_are_correct = True
        directories = [self._train_dir, self._test_dir, self._validation_dir]
        print(f'Checking {directories} for invalid files')
        for directory in directories:
            sub_dirs = os.listdir(directory)
            pbar = tqdm(total=len(list(
                itertools.chain.from_iterable([os.listdir(os.path.join(directory, sub_dir)) for sub_dir in sub_dirs]))),
                        file=sys.stdout)
            for sub_dir in sub_dirs:
                for filename in os.listdir(os.path.join(directory, sub_dir)):
                    file_path = os.path.join(directory, sub_dir, filename)
                    try:
                        image.load_img(file_path, target_size=(self.input_shape[0], self.input_shape[1]))
                    except Exception:
                        all_files_are_correct = False
                        print(f'Bad format for: {file_path}')
                    pbar.update(n=1)
            pbar.close()
        return all_files_are_correct

    def _load_generator(self):
        """Initiate generator from the train, test and validation directories"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            shear_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        validation_datagen = ImageDataGenerator(rescale=1. / 255)
        # validation_datagen = ImageDataGenerator(
        #     rescale=1. / 255,
        #     rotation_range=20,
        #     shear_range=0.1,
        #     width_shift_range=0.1,
        #     height_shift_range=0.1,
        #     zoom_range=0.1,
        #     horizontal_flip=True,
        #     fill_mode='nearest'
        # )
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        # test_datagen = ImageDataGenerator(
        #     rescale=1. / 255,
        #     rotation_range=20,
        #     shear_range=0.1,
        #     width_shift_range=0.1,
        #     height_shift_range=0.1,
        #     zoom_range=0.1,
        #     horizontal_flip=True,
        #     fill_mode='nearest'
        # )

        self.train_generator = train_datagen.flow_from_directory(
            self._train_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            class_mode='categorical')
        self.test_generator = test_datagen.flow_from_directory(
            self._test_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            class_mode='categorical')
        self.validation_generator = validation_datagen.flow_from_directory(
            self._validation_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            class_mode='categorical')
        self.train_set_length = self.train_generator.samples
        self.validation_set_length = self.validation_generator.samples
        self.test_set_length = self.test_generator.samples

    def reload_generator(self):
        """Reload generator in case we need the data after the generator is empty"""
        self._load_generator()

    def get_classes(self):
        """Gets the classes handle by the generators"""
        return self.train_generator.class_indices
