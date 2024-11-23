from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from PIL.Image import Image as img
from ._c_types import NdArrayLike
import numpy as np
import random

BRIGHTNESS_UP_FACTOR = 1.5
BRIGHTNESS_DOWN_FACTOR = 0.75

def create(a:NdArrayLike, /) -> img:
    return Image.fromarray(a)

def to_numpy(a:img, /) -> NdArrayLike:
    return np.array(a)

def crop(a:NdArrayLike, /, **kw) -> img:
    assert "box" in kw

    _box = kw["box"]
    _img = create(a)
    return _img.crop(_box)

def border_crop_and_resize(a:NdArrayLike, /, **kw) -> img:
    assert "border" in kw

    height = a.shape[0]
    width  = a.shape[1]

    _border = kw["border"]
    _img    = create(a)
    return to_numpy(ImageOps.crop(_img, _border).resize((width, height)))

def crop_and_resize(a:NdArrayLike, /, **kw) -> NdArrayLike:
    height = a.shape[0]
    width  = a.shape[1]
    
    _img = crop(a, kw)
    return to_numpy(_img.resize((width, height)))

def rotate(a:NdArrayLike, /, **kw): # does not behave like intended !
    assert "theta" in kw
    _img = create(a)
    return to_numpy(_img.rotate(kw["theta"], expand=True))

def random_rotate(a:NdArrayLike, /, **_):
    _theta = random.random() * 360.0
    return rotate(a, theta=_theta)

def transpose(a:NdArrayLike, /, **_):
    _img = create(a)
    return to_numpy(_img.transpose(Image.Transpose.TRANSPOSE))

def flip(a:NdArrayLike, /, **_):
    _img = create(a)
    return to_numpy(ImageOps.flip(_img))

def blurer(a:NdArrayLike, /, **_):
    _img = create(a)
    return to_numpy(_img.filter(ImageFilter.BLUR))

def sharper(a:NdArrayLike, /, **_):
    _img = create(a)
    return to_numpy(_img.filter(ImageFilter.SHARPEN))

def smoother(a:NdArrayLike, /, **_):
    _img = create(a)
    return to_numpy(_img.filter(ImageFilter.SMOOTH))

def lighter(a:NdArrayLike, /, **_):
    _img = create(a)
    enhancer = ImageEnhance.Brightness(_img)
    return to_numpy(enhancer.enhance(BRIGHTNESS_UP_FACTOR))

def darker(a:NdArrayLike, /, **_):
    _img = create(a)
    enhancer = ImageEnhance.Brightness(_img)
    return to_numpy(enhancer.enhance(BRIGHTNESS_DOWN_FACTOR))

def contrast(a:NdArrayLike, /, **kw):
    assert "contrast_factor" in kw

    _img = create(a)
    enhancer = ImageEnhance.Contrast(_img)
    return to_numpy(enhancer.enhance(kw["contrast_factor"]))

def brightness(a:NdArrayLike, /, **kw):
    assert "brightness_factor" in kw

    _img = create(a)
    enhancer = ImageEnhance.Brightness(_img)
    return to_numpy(enhancer.enhance(kw["brightness_factor"]))

def sharpness(a:NdArrayLike, /, **kw):
    assert "sharpness_factor" in kw

    _img = create(a)
    enhancer = ImageEnhance.Sharpness(_img)
    return to_numpy(enhancer.enhance(kw["sharpness_factor"]))