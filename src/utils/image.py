from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from PIL.Image import Image as img
from ._c_types import NdArrayLike
import numpy as np
import random

BRIGTHNESS_UP_FACTOR = 1.5
BRIGTHNESS_DOWN_FACTOR = 0.75

def create(a:NdArrayLike, /) -> img:
    return Image.fromarray(a)

def to_numpy(a:img, /) -> NdArrayLike:
    return np.array(a)

def border_crop(a:NdArrayLike, /, **kw) -> img:
    assert "border" in kw
    _border = kw["border"]
    _img    = create(a)
    return ImageOps.crop(_img, _border)

def general_crop(a:NdArrayLike, /, **kw) -> img:
    assert "box" in kw
    _box = kw["box"]
    _img = create(a)
    return _img.crop(_box)

def random_crop(a:NdArrayLike, /, **kw) -> img:
    assert "box_shape" in kw
    assert "min_size" in kw

    _box_left  = random.randint(kw["box_shape"][0], kw["box_shape"][2]-kw["min_size"])
    _box_upper = random.randint(kw["box_shape"][1], kw["box_shape"][3]-kw["min_size"])
    _box_right = random.randint(_box_left,  kw["box_shape"][2])
    _box_lower = random.randint(_box_upper, kw["box_shape"][3])

    return general_crop(a, box=(_box_left, _box_upper, _box_right, _box_lower))

def crop_and_resize(a:NdArrayLike, /, **kw) -> NdArrayLike:
    height = a.shape[0]
    width  = a.shape[1]
    
    _img = general_crop(a, kw)
    return to_numpy(_img.resize((width, height)))

def rotate(a:NdArrayLike, /, **kw):
    assert "theta" in kw
    _img = create(a)
    return to_numpy(_img.rotate(kw["theta"]))

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
    return to_numpy(enhancer.enhance(BRIGTHNESS_UP_FACTOR))

def darker(a:NdArrayLike, /, **_):
    _img = create(a)
    enhancer = ImageEnhance.Brightness(_img)
    return to_numpy(enhancer.enhance(BRIGTHNESS_DOWN_FACTOR))

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