import random
import math
import torch
from PIL import Image

def resize(img, boxes, size, max_size=1000):
    ## **warning
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w, h)
        size_max = max(w, h)
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(max_size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h
    return img.resize((ow, oh), Image.BILINEAR), boxes * torch.Tensor([sw, sh, sw, sh])


def random_crop(img, boxes):
    success = False
    for attempt in range(10):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(0.56, 1.0) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w
        
        if w <= img.size[0] and h <= img.size[1]:
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            success = True
            break

    # Fallback
    if not success:
        w = h = min(img.size[0], img.size[1])
        x = (img.size[0] - w) // 2
        y = (img.size[1] - h) // 2

    img = img.crop((x, y, x + w, y + h))
    boxes -= torch.Tensor([x, y, x, y])
    boxes[:, 0::2].clamp_(min=0, max=w-1)
    boxes[:, 1::2].clamp_(min=0, max=h-1)
    return img, boxes


def center_crop(img, boxes, size):
    w, h = img.size
    ow, oh = size
    i = int(round((h - oh) / 2.))
    j = int(round((w - ow) / 2.))
    img = img.crop((j, i, j + ow, i + oh))
    boxes -= torch.Tensor([j, i, j, i])
    boxes[:, 0::2].clamp_(min=0, max=ow-1)
    boxes[:, 1::2].clamp_(min=0, max=oh-1)
    return img, boxes


def random_flip(img, boxes):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w = img.width
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 0] = xmin
        boxes[:, 2] = xmax
    return img, boxes