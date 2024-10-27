import os
import random
import shutil

file_list = []

directory = os.path.join('unused', 'leftover', 'hats', 'temp')

file_list += os.listdir(directory)

random.shuffle(file_list)

dest = os.path.join('dataset', 'clothes')
for index, file in enumerate(file_list):
    if index < 3500:
        if not os.path.exists(os.path.join(dest, 'train', 'hats')):
            os.mkdir(os.path.join(dest, 'train', 'hats'))
        shutil.move(os.path.join(directory, file), os.path.join(dest, 'train', 'hats', f'{index}.{file.split(".")[-1]}'))
    elif index < 4250:
        if not os.path.exists(os.path.join(dest, 'test', 'hats')):
            os.mkdir(os.path.join(dest, 'test', 'hats'))
        shutil.move(os.path.join(directory, file),
                    os.path.join(dest, 'test', 'hats', f'{index}.{file.split(".")[-1]}'))
    elif index < 5000:
        if not os.path.exists(os.path.join(dest, 'validation', 'hats')):
            os.mkdir(os.path.join(dest, 'validation', 'hats'))
        shutil.move(os.path.join(directory, file),
                    os.path.join(dest, 'validation', 'hats', f'{index}.{file.split(".")[-1]}'))