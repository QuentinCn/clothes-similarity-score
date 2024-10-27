import os.path
import random
import shutil

src_dir = 'color'
dest_dir = os.path.join('dataset', 'color1')

if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

if not os.path.exists(os.path.join(dest_dir, 'train')):
    os.mkdir(os.path.join(dest_dir, 'train'))
if not os.path.exists(os.path.join(dest_dir, 'test')):
    os.mkdir(os.path.join(dest_dir, 'test'))
if not os.path.exists(os.path.join(dest_dir, 'validation')):
    os.mkdir(os.path.join(dest_dir, 'validation'))

for subdir in os.listdir(src_dir):
    if not os.path.exists(os.path.join(dest_dir, 'train', subdir)):
        os.mkdir(os.path.join(dest_dir, 'train', subdir))
    if not os.path.exists(os.path.join(dest_dir, 'test', subdir)):
        os.mkdir(os.path.join(dest_dir, 'test', subdir))
    if not os.path.exists(os.path.join(dest_dir, 'validation', subdir)):
        os.mkdir(os.path.join(dest_dir, 'validation', subdir))

    files = os.listdir(os.path.join(src_dir, subdir))
    random.shuffle(files)
    for i in range(1000):
        file = files.pop()
        shutil.copy(os.path.join(src_dir, subdir, file), os.path.join(dest_dir, 'train', subdir, file))
    for i in range(300):
        file = files.pop()
        shutil.copy(os.path.join(src_dir, subdir, file), os.path.join(dest_dir, 'validation', subdir, file))
    for i in range(300):
        file = files.pop()
        shutil.copy(os.path.join(src_dir, subdir, file), os.path.join(dest_dir, 'test', subdir, file))
