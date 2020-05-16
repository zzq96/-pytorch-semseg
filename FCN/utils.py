import os
import torch
import random
import numpy as np
import PIL
from PIL import Image
import numbers
import torchvision
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import functional as F
class VOCSegmentation(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 root,
                 year='2012',
                 image_set='train', crop_size = None, use=1):
        #super(VOCSegmentation, self).__init__(root, transforms, transform, target_transform)
        self.crop_size = crop_size
        if crop_size:
            self.crop = MultiRandomCrop(crop_size)
        self.To_Tensor = torchvision.transforms.ToTensor()
        self.year = year
        self.image_set = image_set
        self.root = root
        voc_root = os.path.join(self.root, "VOC" + year)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')
        print(voc_root)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]


        self.images = self._filter(self.images)
        self.masks = self._filter(self.masks)

        length = len(self.images)
        self.images = self.images[:length//use]
        self.masks = self.masks[:length//use]

        assert (len(self.images) == len(self.masks))
    
    def _filter(self, images):
        return [im for im in images if (Image.open(im).size[0]>=self.crop_size[1] and \
            Image.open(im).size[1]>=self.crop_size[0])]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        img = Image.open(self.images[index])#RGB模式
        target = Image.open(self.masks[index])#P模式8位调色板模式, 像素值已经是对应类别的编号了
        if 'crop' in dir(self):
            if img.size[1] < self.crop_size[0] or img.size[0] < self.crop_size[1]:#PIL图像的size长宽是反的！
                print("resize")
                img, target= img.resize((self.crop_size[1],self.crop_size[0])), target.resize((self.crop_size[1],self.crop_size[0]))
            else:
                img, target = self.crop(img, target)

        # img, target = self.To_Tensor(img), self.To_Tensor(target)
        img, target = self.To_Tensor(img), torch.from_numpy(np.array(target))
        target[target==255] = 0

        return torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img), target.long()

    def __len__(self):
        return len(self.images)

class MultiRandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, *imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        result = []
        i, j, h, w = self.get_params(imgs[0], self.size)
        for img in imgs:
            if self.padding is not None:
                img = F.pad(img, self.padding, self.fill, self.padding_mode)

            # pad the width if needed
            if self.pad_if_needed and img.size[0] < self.size[1]:
                img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            # pad the height if needed
            if self.pad_if_needed and img.size[1] < self.size[0]:
                img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            result.append(F.crop(img, i, j, h, w))
        return result

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)