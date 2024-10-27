import os

path=os.path.join('dataset', 'symboles')
dirs=os.listdir(path)

for subdir in os.listdir(os.path.join(path,dirs[0])):
    for middir in dirs:
        nb_files = len(os.listdir(os.path.join(path,middir,subdir)))
        if nb_files < 100:
            print(f'{nb_files} files in {os.path.join(path,middir,subdir)}')