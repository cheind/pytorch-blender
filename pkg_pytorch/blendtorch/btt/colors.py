import numpy as np


def gamma(x, coeff=2.2):
    """Return sRGB (gamme encoded o=i**(1/coeff)) image.

    This gamma encodes linear colorspaces as produced by Blender
    renderings.

    Params
    ------
    x: HxWxC uint8 array
        C either 3 (RGB) or 4 (RGBA)
    coeff: scalar
        correction coefficient

    Returns
    -------
    y: HxWxC uint8 array
        Gamma encoded array.
    """
    y = x[..., :3].astype(np.float32) / 255
    y = np.uint8(255.0 * y ** (1 / 2.2))
    if x.shape[-1] == 3:
        return y
    else:
        return np.concatenate((y, x[..., 3:4]), axis=-1)
