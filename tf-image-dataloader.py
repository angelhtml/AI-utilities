import tensorflow as tf
import numpy as np


train_label = [1]
train_image = ["dog.jpg"]


def shuffle_generator(image, label, seed, grayscale):
    idx = np.arange(len(image))
    np.random.default_rng(seed).shuffle(idx)
    for i in idx:
        img = tf.keras.utils.load_img(
            image[i],
            grayscale=grayscale,
            color_mode="rgb",
            target_size=(512, 512),
            interpolation="nearest",
            keep_aspect_ratio=False,
        )
        img = tf.keras.utils.img_to_array(img, data_format=None, dtype=None)

        yield img, label[i]


dataset = tf.data.Dataset.from_generator(
    shuffle_generator,
    args=[train_image, train_label, 42, False],
    output_signature=(tf.TensorSpec(shape=(512, 512, 3)), tf.TensorSpec(shape=())),
)

for data in dataset.batch(1):
    print(data)
