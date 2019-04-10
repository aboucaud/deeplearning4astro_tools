import numpy as np


def plot_results(model, img, y_true):
    import matplotlib.pyplot as plt

    y_pred = model.predict(np.expand_dims(img, 0))
    y_true = np.expand_dims(y_true, 0).round()
    
    img = np.squeeze(img)
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    fig_size = (8, 8)
    fig, ax = plt.subplots(nrows=2, ncols=2,
                           sharex=True, sharey=True, figsize=fig_size)
    ax[0, 0].imshow(img, origin='lower')
    ax[0, 1].imshow(y_true, origin='lower')
    ax[1, 1].imshow(y_pred.round(), origin='lower')
    
    for axes in ax.flat:
        axes.axis('off')


def plot_random_results(model, X_test, y_test, n_gal=5):
    import matplotlib.pyplot as plt
    idx = np.random.randint(0, len(y_test), size=n_gal)
    X = X_test[idx]
    if X.ndim == 3:
        X = np.expand_dims(X, -1)
    y_true = y_test[idx]
    y_pred = model.predict(X)

    titles = [
        'blend',
        'true segmentation',
        'output',
        'output thresholded',
    ]
    fig_size = (10, 12)
    fig, ax = plt.subplots(nrows=n_gal, ncols=4, figsize=fig_size)
    for i in range(n_gal):
        img = np.squeeze(X[i])
        yt = np.squeeze(y_true[i])
        yp = np.squeeze(y_pred[i])
        ax[i, 0].imshow(img)
        ax[i, 1].imshow(yt)
        ax[i, 2].imshow(yp)
        ax[i, 3].imshow(yp.round())
        if i == 0:
            for idx, a in enumerate(ax[i]):
                a.set_title(titles[idx])
        for a in ax[i]:
            a.set_axis_off()

def plot_history(history):
    import matplotlib.pyplot as plt
    plt.semilogy(history.epoch, history.history['loss'], label='loss')
    plt.semilogy(history.epoch, history.history['val_loss'], label='val_loss')
    plt.title('Training performance')
    plt.xlim(1, None)
    plt.legend()