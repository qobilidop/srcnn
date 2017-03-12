import h5py
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import list_pictures
from scipy.misc import imresize

from toolbox.paths import data_dir


def load_data(train_dir='Train', test_dir='Test/Set5', channel=0):
    x_train, y_train = generate(train_dir, channel=channel)
    x_test, y_test = generate(test_dir, save_path='test.h5', stride=21,
                              channel=channel)
    return (x_train, y_train), (x_test, y_test)


def generate(directory, save_path='train.h5', size_input=33, size_label=21,
             scale=3, stride=14, channel=0):
    # settings
    directory = data_dir / directory

    # initialization
    data = []
    label = []
    padding = abs(size_input - size_label) // 2

    # generate data
    filepaths = list_pictures(str(directory), ext='bmp')

    for path in filepaths:
        image = load_img(path)
        image = image.convert('YCbCr')
        image = img_to_array(image)

        im_label = modcrop(image, scale)
        hei, wid = im_label.shape[:2]
        im_input = imresize(imresize(im_label, 1 / scale, 'bicubic'),
                            (hei, wid), 'bicubic')

        for x in range(0, hei - size_input + 1, stride):
            for y in range(0, wid - size_input + 1, stride):
                subim_input = im_input[x: x + size_input, y: y + size_input]
                subim_label = im_label[x + padding: x + padding + size_label,
                              y + padding: y + padding + size_label]
                data += [subim_input]
                label += [subim_label]

    order = np.random.permutation(len(data))
    data = np.stack(data)[order]
    label = np.stack(label)[order]

    with h5py.File(save_path, 'w') as f:
        f.create_dataset('dat', data=data)
        f.create_dataset('lab', data=label)

    if channel == 'all':
        return data, label
    else:
        return data[:, :, :, channel:channel + 1], \
               label[:, :, :, channel:channel + 1]


def modcrop(image, scale):
    size = np.array(image.shape)
    size -= size % scale
    image = image[:size[0], :size[1]]
    return image
