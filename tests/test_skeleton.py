__author__ = "Martin Becker"
__copyright__ = "Martin Becker"
__license__ = "MIT"


from numpy import dtype


def test_tensorflow():

    import numpy as np
    import tensorflow as tf

    (x_train, y_train) = \
        np.random.random((100, 32, 32, 3)), \
        np.random.randint(0, 100, (100, 1))

    print("Load model")
    model = tf.keras.applications.EfficientNetB0(
        include_top=True,
        weights=None,
        input_shape=(32, 32, 3),
        classes=100,)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=1, batch_size=64)


if __name__ == '__main__':
    test_tensorflow()