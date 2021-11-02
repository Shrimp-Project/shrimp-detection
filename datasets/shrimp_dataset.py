import os
import numpy as np
import cv2

import torch
import torchvision.transforms as transforms

from PIL import Image
from datasets.encoder import DataEncoder
from datasets.transform import resize, center_crop, random_flip

import xml.etree.ElementTree as ET

class ShrimpDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, image_scales, get_segmentation=False):
        self.category = ['shrimp']
        self.img_root_dir = os.path.join(dataset_dir, 'imgs')
        self.img_list = os.path.join(dataset_dir, 'image_list.txt')
        self.annotations_dir = os.path.join(dataset_dir, 'annotations')
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
        self.train = True
        self.input_size = image_scales

        self.fnames = []
        self.boxes = []
        self.labels = []
        self.num_samples = 0
        self.get_img_annotations()
        self.get_segmentation = get_segmentation

        self.encoder = DataEncoder(self.input_size)

    def get_img_annotations(self):
        with open(self.img_list) as f:
            lines = f.readlines()
        self.num_samples = len(lines)

        for line in lines:
            splited = line.strip()
            self.fnames.append(splited + '.jpg')
            box = []
            label = []
            ann = os.path.join(self.annotations_dir, splited + '.xml')
            rec = parse_rec(ann)
            for r in rec:
                box.append(r['bbox'])
                label.append(self.category.index(r['name']))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.img_root_dir, fname))

        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Denoise image
        img = self.denoise(img)

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        size = self.input_size

        # data augmentation
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = resize(img, boxes, size)
        else:
            img, boxes = resize(img, boxes, size)
            img, boxes = center_crop(img, boxes, size) ##**

        if self.get_segmentation:
            img = self.segmentation(img)

        # Normalize
        img = self.transform(img)
        return img, boxes, labels

    def __len__(self):
        return self.num_samples

    def denoise(self, img):
        # convert PIL image to cv2 image
        img = np.array(img)

        # denoise
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # convert cv2 image to PIL image
        img = Image.fromarray(img)
        return img

    def segmentation(self, img):
        # convert PIL image to cv2 image
        img = np.array(img)

        # init input for kmean
        twoDimage = img.reshape((-1, 3))
        twoDimage = np.float32(twoDimage)

        # init parameter for kmean
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 2
        attempts = 10

        # kmean
        ret, label, center = cv2.kmeans(twoDimage, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((img.shape))
        
        # convert cv2 image to PIL image
        result_image = Image.fromarray(result_image)

        return result_image

    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = self.input_size[1]
        w = self.input_size[0]
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w, h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)


def parse_rec(filename):
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                            int(bbox.find('ymin').text),
                            int(bbox.find('xmax').text),
                            int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects