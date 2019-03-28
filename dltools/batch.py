import numpy as np


class BatchGeneratorBuilder(object):
    """A batch generator builder for generating batches of images on the fly.

    This class is a way to build training and
    validation generators that yield each time a tuple (X, y) of mini-batches.
    The generators are built in a way to fit into keras API of `fit_generator`
    (see https://keras.io/models/model/).

    The fit function from `Classifier` should then use the instance
    to build train and validation generators, using the method
    `get_train_valid_generators`

    Parameters
    ==========

    X_array : ArrayContainer of int
        vector of image data to train on
    y_array : vector of int
        vector of object labels corresponding to `X_array`

    """
    def __init__(self, X_array, y_array):
        self.X_array = X_array
        self.y_array = y_array
        self.nb_examples = len(X_array)

    def get_train_valid_generators(self, batch_size=256, valid_ratio=0.1):
        """Build train and valid generators for keras.

        This method is used by the user defined `Classifier` to o build train
        and valid generators that will be used in keras `fit_generator`.

        Parameters
        ==========

        batch_size : int
            size of mini-batches
        valid_ratio : float between 0 and 1
            ratio of validation data

        Returns
        =======

        a 4-tuple (gen_train, gen_valid, nb_train, nb_valid) where:
            - gen_train is a generator function for training data
            - gen_valid is a generator function for valid data
            - nb_train is the number of training examples
            - nb_valid is the number of validation examples
        The number of training and validation data are necessary
        so that we can use the keras method `fit_generator`.
        """
        nb_valid = int(valid_ratio * self.nb_examples)
        nb_train = self.nb_examples - nb_valid
        indices = np.arange(self.nb_examples)
        train_indices = indices[0:nb_train]
        valid_indices = indices[nb_train:]
        gen_train = self._get_generator(
            indices=train_indices, batch_size=batch_size)
        gen_valid = self._get_generator(
            indices=valid_indices, batch_size=batch_size)
        return gen_train, gen_valid, nb_train, nb_valid

    def _get_generator(self, indices=None, batch_size=256):
        if indices is None:
            indices = np.arange(self.nb_examples)
        # Infinite loop, as required by keras `fit_generator`.
        # However, as we provide the number of examples per epoch
        # and the user specifies the total number of epochs, it will
        # be able to end.
        while True:
            X = self.X_array[indices]
            y = self.y_array[indices]

            # converting to float needed?
            X = np.array(X, dtype='float32')
            y = np.array(y, dtype='float32')

            # Yielding mini-batches
            for i in range(0, len(X), batch_size):

                X_batch = [np.expand_dims(img, -1)
                           for img in X[i:i + batch_size]]
                y_batch = [np.expand_dims(seg, -1)
                           for seg in y[i:i + batch_size]]

                for j in range(len(X_batch)):

                    # flip images
                    if np.random.randint(2):
                        X_batch[j] = np.flip(X_batch[j], axis=0)
                        y_batch[j] = np.flip(y_batch[j], axis=0)

                    if np.random.randint(2):
                        X_batch[j] = np.flip(X_batch[j], axis=1)
                        y_batch[j] = np.flip(y_batch[j], axis=1)

                    # TODO add different data augmentation steps

                yield np.array(X_batch), np.array(y_batch)
