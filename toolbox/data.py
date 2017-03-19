import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from toolbox.image import bicubic_resize
from toolbox.image import modcrop
from toolbox.paths import data_dir


def load_set(name, sub_size=7, sub_stride=3, scale=3):
    dataset_dir = data_dir / name
    lr_sub_arrays = []
    hr_sub_arrays = []
    for path in dataset_dir.glob('*'):
        lr_image, hr_image = load_image_pair(str(path), scale=scale)
        gen_sub = generate_sub_images
        lr_sub_arrays += [
            img_to_array(img)
            for img in gen_sub(lr_image, sub_size, sub_stride)
        ]
        hr_sub_arrays += [
            img_to_array(img)
            for img in gen_sub(hr_image, sub_size * scale, sub_stride * scale)
        ]
    x = np.stack(lr_sub_arrays)
    y = np.stack(hr_sub_arrays)
    return x, y


def load_image_pair(path, scale=3):
    image = load_img(path)
    image = image.convert('YCbCr')
    hr_image = modcrop(image, scale)
    lr_image = bicubic_resize(hr_image, 1 / scale)
    return lr_image, hr_image


def generate_sub_images(image, crop_size, stride):
    for i in range(0, image.size[0] - crop_size + 1, stride):
        for j in range(0, image.size[1] - crop_size + 1, stride):
            yield image.crop([i, j, i + crop_size, j + crop_size])
