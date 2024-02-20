# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys
import io
import imgviz
import labelme
import PIL.Image
from labelme import PY2
from labelme import QT4


def apply_exif_orientation(image):
    try:
        exif = image._getexif()
    except AttributeError:
        exif = None

    if exif is None:
        return image

    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in exif.items()
        if k in PIL.ExifTags.TAGS
    }

    orientation = exif.get("Orientation", None)

    if orientation == 1:
        # do nothing
        return image
    elif orientation == 2:
        # left-to-right mirror
        return PIL.ImageOps.mirror(image)
    elif orientation == 3:
        # rotate 180
        return image.transpose(PIL.Image.ROTATE_180)
    elif orientation == 4:
        # top-to-bottom mirror
        return PIL.ImageOps.flip(image)
    elif orientation == 5:
        # top-to-left mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
    elif orientation == 6:
        # rotate 270
        return image.transpose(PIL.Image.ROTATE_270)
    elif orientation == 7:
        # top-to-right mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
    elif orientation == 8:
        # rotate 90
        return image.transpose(PIL.Image.ROTATE_90)
    else:
        return image


def load_image_file(filename):
    try:
        image_pil = PIL.Image.open(filename)
    except IOError:
        print('error')
        return

    # apply orientation to image according to exif
    image_pil = apply_exif_orientation(image_pil)

    with io.BytesIO() as f:
        ext = osp.splitext(filename)[1].lower()
        if PY2 and QT4:
            format = "PNG"
        elif ext in [".jpg", ".jpeg"]:
            format = "JPEG"
        else:
            format = "PNG"
        image_pil.save(f, format=format)
        f.seek(0)
        return f.read()

