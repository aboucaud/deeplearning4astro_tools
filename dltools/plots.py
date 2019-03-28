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