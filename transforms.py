from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import random
import math
from skimage import io, transform
import torch
import numpy as np
from torchvision import transforms, utils
from torchvision.transforms import *
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label, index , pid, cid = sample['image'], sample['label'], sample['index'], sample['id'], sample['camera']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        #landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'label': label, 'index':index, 'id': pid, 'camera': cid}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label, index , pid, cid = sample['image'], sample['label'], sample['index'], sample['id'], sample['camera']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

      
        return {'image': image, 'label': label, 'index':index, 'id': pid, 'camera': cid}

class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.
    Args:
    - height (int): target image height.
    - width (int): target image width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, sample):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        image, label, index , pid, cid = sample['image'], sample['label'], sample['index'], sample['id'], sample['camera']
        if random.uniform(0, 1) > self.p:
            image =  image.resize((self.width, self.height), self.interpolation)
            return {'image': image, 'label': label, 'index':index, 'id': pid, 'camera': cid}

        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = image.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return {'image': cropped_img, 'label': label, 'index':index, 'id': pid, 'camera': cid}


class ColorAugmentation(object):
    """
    Randomly alter the intensities of RGB channels
    Reference:
    Krizhevsky et al. ImageNet Classification with Deep ConvolutionalNeural Networks. NIPS 2012.
    """

    def __init__(self, p=0.5):
        self.p = p
        self.eig_vec = torch.Tensor([
            [0.4009, 0.7192, -0.5675],
            [-0.8140, -0.0045, -0.5808],
            [0.4203, -0.6948, -0.5836],
        ])
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def _check_input(self, tensor):
        assert tensor.dim() == 3 and tensor.size(0) == 3

    def __call__(self, sample):
        image, label, index , pid, cid = sample['image'], sample['label'], sample['index'], sample['id'], sample['camera']
        if random.uniform(0, 1) > self.p:
            return {'image': image, 'label': label, 'index':index, 'id': pid, 'camera': cid}
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        image = image + quatity.view(3, 1, 1)
        return {'image': image, 'label': label, 'index':index, 'id': pid, 'camera': cid}

class RandomErasing(object):

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, sample):
        image, label, index , pid, cid = sample['image'], sample['label'], sample['index'], sample['id'], sample['camera']
        if random.uniform(0, 1) > self.probability:
            return {'image': image, 'label': label, 'index':index, 'id': pid, 'camera': cid}

        for attempt in range(100):
            area = image.size()[1] * image.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < image.size()[2] and h < image.size()[1]:
                x1 = random.randint(0, image.size()[1] - h)
                y1 = random.randint(0, image.size()[2] - w)
                if image.size()[0] == 3:
                    image[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    image[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    image[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    image[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return {'image': image, 'label': label, 'index':index, 'id': pid, 'camera': cid}

        return {'image': image, 'label': label, 'index':index, 'id': pid, 'camera': cid}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, index , pid, cid = sample['image'], sample['label'], sample['index'], sample['id'], sample['camera']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': label, 
                'index':index,
                'id': pid, 
                'camera': cid}

def data_transforms(height, width, random_erase, color_jitter, color_aug):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = Normalize(mean=imagenet_mean, std=imagenet_std)
    transform  = []
    transform += [Random2DTranslation(height, width)]
    transform += [RandomHorizontalFlip()]
    if color_jitter:
        transform += [ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)]
    
    if color_aug:
        transform += [ColorAugmentation()]
    transform += [normalize]
    if random_erase:
        transform += [RandomErasing()]
    transform += [ToTensor()]
    transform = Compose(transform)
    return transform
