I loaded VGG19 feature extractor, pretrained on ImageNet. For each image
from COCO 2017 train dataset I got activations of a top layer.

I made dictionary with image paths as keys and activations as values.

In search.py I just compute activations of the same layer for new image and
compare with all values in dictionary with L2 norm.

python3 src/generate.py path (path - path to coco dataset)

python3 src/search.py path n (path - path to image, n - num of most similar images)