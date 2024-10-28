import os

from PIL import Image

dir_path = 'color'

if not os.path.exists(dir_path):
    os.mkdir(dir_path)

im = Image.new('RGB', (224, 224))

red_path = 'red'
green_path = 'green'
blue_path = 'blue'
black_path = 'black'
white_path = 'white'
cyan_path = 'cyan'
yellow_path = 'yellow'
orange_path = 'orange'
purple_path = 'purple'
grey_path = 'grey'
pink_path = 'pink'
brown_path = 'brown'

if not os.path.exists(os.path.join(dir_path, red_path)):
    os.mkdir(os.path.join(dir_path, red_path))
if not os.path.exists(os.path.join(dir_path, blue_path)):
    os.mkdir(os.path.join(dir_path, blue_path))
if not os.path.exists(os.path.join(dir_path, green_path)):
    os.mkdir(os.path.join(dir_path, green_path))
if not os.path.exists(os.path.join(dir_path, white_path)):
    os.mkdir(os.path.join(dir_path, white_path))
if not os.path.exists(os.path.join(dir_path, black_path)):
    os.mkdir(os.path.join(dir_path, black_path))
if not os.path.exists(os.path.join(dir_path, cyan_path)):
    os.mkdir(os.path.join(dir_path, cyan_path))
if not os.path.exists(os.path.join(dir_path, yellow_path)):
    os.mkdir(os.path.join(dir_path, yellow_path))
if not os.path.exists(os.path.join(dir_path, orange_path)):
    os.mkdir(os.path.join(dir_path, orange_path))
if not os.path.exists(os.path.join(dir_path, purple_path)):
    os.mkdir(os.path.join(dir_path, purple_path))
if not os.path.exists(os.path.join(dir_path, grey_path)):
    os.mkdir(os.path.join(dir_path, grey_path))
if not os.path.exists(os.path.join(dir_path, pink_path)):
    os.mkdir(os.path.join(dir_path, pink_path))
if not os.path.exists(os.path.join(dir_path, brown_path)):
    os.mkdir(os.path.join(dir_path, brown_path))


def create_img(r, g, b, color_path):
    data = [(r, g, b) for y in range(im.size[1]) for x in range(im.size[0])]
    im.putdata(data)
    im.save(os.path.join(dir_path, color_path, f'{r}_{g}_{b}.png'))


g_offset = 4
w_offset = 1

r = 0
g = 0
b = 0
while r < 256:
    g = 0
    while g < 256:
        b = 0
        while b < 256:
            if r > 240 and g > 240 and b > 240:
                create_img(r, g, b, white_path)
            elif r < 50 and g < 50 and b < 50:
                create_img(r, g, b, black_path)
            elif abs(r - g) < 15 and abs(g - b) < 15 and abs(r - b) < 15:
                create_img(r, g, b, grey_path)
            elif r > 200 and g < 100 and b < 100:
                create_img(r, g, b, red_path)
            elif r > 200 and g > 150 and b < 50:
                create_img(r, g, b, orange_path)
            elif r > 200 and g > 200 and b < 100:
                create_img(r, g, b, yellow_path)
            elif g > 150 and r < 100 and b < 100:
                create_img(r, g, b, green_path)
            elif g > 200 and b > 200 and r < 100:
                create_img(r, g, b, cyan_path)
            elif b > 200 and r < 100 and g < 100:
                create_img(r, g, b, blue_path)
            elif r > 150 and b > 150 and g < 100:
                create_img(r, g, b, purple_path)
            elif r > 200 and g < 150 and b > 150:
                create_img(r, g, b, pink_path)
            elif r > 150 and g > 100 and b < 50:
                create_img(r, g, b, brown_path)

            if r > 240 and g > 240 and b > 240:
                b += w_offset
            else:
                b += g_offset
        if r > 240 and g > 240 and b > 240:
            g += w_offset
        else:
            g += g_offset
    if r > 240 and g > 240 and b > 240:
        r += w_offset
    else:
        r += g_offset
