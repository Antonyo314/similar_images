import argparse
import pickle

import keras.backend as K
from keras import models
from keras.applications import vgg19
from keras.preprocessing.image import ImageDataGenerator

K.set_floatx('float16')


def build_model():
    model = vgg19.VGG19(weights='imagenet',
                        include_top=False)

    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    content_layer = 'block5_pool'
    layer_features = outputs_dict[content_layer]

    activation_model = models.Model(inputs=model.input, outputs=layer_features)
    return activation_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    coco_dir_path = args.path

    generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(coco_dir_path,
                                                                         target_size=(150, 150),
                                                                         batch_size=32,
                                                                         shuffle=False,
                                                                         class_mode='binary')
    model = build_model()

    activations = model.predict_generator(generator, verbose=1)
    activations = [activation.reshape(-1) for activation in activations]

    filenames = [filename.split('/')[-1] for filename in generator.filenames]

    img_activation = dict(zip(filenames, activations))

    with open('dict.pkl', 'wb') as f:
        pickle.dump(img_activation, f, pickle.HIGHEST_PROTOCOL)
