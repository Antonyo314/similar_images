import argparse
import pickle

import numpy as np
from generate import build_model
from keras.preprocessing import image


def distance(img_1, img_2):
    return np.sum(np.square(img_1 - img_2))


def find_most_similar(target_value, n):
    keys = list(img_activation.keys())
    values = np.array(list(img_activation.values()))
    dist = [distance(v, target_value) for v in values]
    arg = np.argsort(dist)
    answer = [keys[a] for a in arg]
    return answer[:n]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('n', type=int)

    args = parser.parse_args()

    with open('dict.pkl', 'rb') as f:
        img_activation = pickle.load(f)

    img_path = args.path
    n = args.n

    img = image.load_img(img_path, target_size=(150, 150))

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    img /= 255.

    model = build_model()

    target_value = model.predict(img)
    target_value = target_value.reshape(-1)

    print(find_most_similar(target_value, n))
