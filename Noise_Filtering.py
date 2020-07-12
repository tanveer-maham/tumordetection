import numpy as np
import dippykit as dip
import warnings
import scipy as sp
import math as mt
import cv2

from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.measure import compare_psnr
from skimage.restoration import denoise_wavelet, cycle_spin
from skimage.util import random_noise


#------------------------------------------------------------------------#


#X is input image, algo is to be sent
def NoiseRemoval(X, algo = 0):
    # AlgoToUse

    if algo == 0:
        # Applying the mean filter
        X_g = sp.ndimage.gaussian_filter(X, 1)
        X_2 = X_g

    if algo == 1:
        # Applying the Median filter
        X_m = sp.ndimage.median_filter(X, 3)
        X_2 = X_m

    if algo == 2:
        # Applying the anistropic filter
        X_an = anisodiff(X, niter=5, kappa=100, gamma=0.15, step=(1., 1.), option=1, ploton=False)
        X_2 = X_an

    if algo == 3:
        # Applying nonlocal means
        X_clip = X[30:180, 150:300]
        sigma_est = np.mean(estimate_sigma(X_clip, multichannel=False))
        print("estimated noise standard deviation = {}".format(sigma_est))
        patch_kw = dict(patch_size=5,  # 5x5 patches
                        patch_distance=6,  # 13x13 search area
                        multichannel=True)
        X_non = denoise_nl_means(X, h=0.8 * sigma_est, fast_mode=True, **patch_kw)

        X_2 = X_non

    if algo == 4:
        # Applying wavelet based
        # Repeat denosing with different amounts of cycle spinning.  e.g.
        # max_shift = 0 -> no cycle spinning
        # max_shift = 1 -> shifts of (0, 1) along each axis
        # max_shift = 3 -> shifts of (0, 1, 2, 3) along each axis
        # etc...
        X = X / 255
        denoise_kwargs = dict(multichannel=False, convert2ycbcr=True, wavelet='db1')
        max_shifts = [0, 1, 3, 5]
        s = 3
        X_w = cycle_spin(X, func=denoise_wavelet, max_shifts=s, func_kw=denoise_kwargs, multichannel=False)

        X_2 = X_w * 255
        X = X * 255

    return X_2

def anisodiff(img, niter=1, kappa=100, gamma=0.1, step=(1., 1.), option=1, ploton=False):

    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep

        fig = pl.figure(figsize=(20, 5.5), num="Anisotropic diffusion")
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

        ax1.imshow(img, interpolation='nearest')
        ih = ax2.imshow(imgout, interpolation='nearest', animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in range(niter):

        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS / kappa) ** 2.) / step[0]
            gE = np.exp(-(deltaE / kappa) ** 2.) / step[1]
        elif option == 2:
            gS = 1. / (1. + (deltaS / kappa) ** 2.) / step[0]
            gE = 1. / (1. + (deltaE / kappa) ** 2.) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma * (NS + EW)

        if ploton:
            iterstring = "Iteration %i" % (ii + 1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)

    return imgout

#------------------------------------------------------------------------#






