import os
from enum import Enum

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .base_dataset import BaseDataset


class ClassNames(Enum):
    ROAD = 0
    SIDEWALK = 1
    CURB = 2
    MANHOLE = 3
    GRASS = 4
    TREES = 5
    VEHICLE = 6
    PERSON = 7
    STATIC_STRUCTURE = 8
    SKY = 9
    # NATURE = 10
    OTHER = 10


class Mappilary(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=11,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        super(Mappilary, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]
        self.label_mapping = {0: ignore_label,
                              1: ignore_label,
                              2: ClassNames.CURB.value,
                              3: ClassNames.STATIC_STRUCTURE.value,
                              4: ClassNames.STATIC_STRUCTURE.value,
                              5: ClassNames.STATIC_STRUCTURE.value,
                              6: ClassNames.STATIC_STRUCTURE.value,
                              7: ClassNames.ROAD.value,
                              8: ClassNames.ROAD.value,
                              9: ClassNames.CURB.value,
                              10: ClassNames.ROAD.value,
                              11: ignore_label,  # ??? pedestrian-area
                              12: ClassNames.ROAD.value,
                              13: ClassNames.ROAD.value,
                              14: ClassNames.ROAD.value,
                              15: ClassNames.SIDEWALK.value,
                              16: ClassNames.STATIC_STRUCTURE.value,
                              17: ClassNames.STATIC_STRUCTURE.value,
                              18: ClassNames.STATIC_STRUCTURE.value,
                              19: ClassNames.PERSON.value,
                              20: ClassNames.PERSON.value,
                              21: ClassNames.PERSON.value,
                              22: ClassNames.PERSON.value,
                              23: ClassNames.ROAD.value,
                              24: ClassNames.ROAD.value,
                              25: ignore_label,
                              26: ClassNames.OTHER.value,
                              27: ClassNames.SKY.value,
                              28: ClassNames.OTHER.value,
                              29: ClassNames.GRASS.value,  # terrain/grass
                              30: ClassNames.TREES.value,  # trees
                              31: ClassNames.OTHER.value,
                              32: ClassNames.STATIC_STRUCTURE.value,
                              33: ClassNames.STATIC_STRUCTURE.value,
                              34: ClassNames.STATIC_STRUCTURE.value,
                              35: ClassNames.STATIC_STRUCTURE.value,
                              36: ClassNames.MANHOLE.value,
                              37: ignore_label,
                              38: ClassNames.STATIC_STRUCTURE.value,
                              39: ClassNames.STATIC_STRUCTURE.value,
                              40: ClassNames.STATIC_STRUCTURE.value,
                              41: ClassNames.MANHOLE.value,
                              42: ClassNames.STATIC_STRUCTURE.value,
                              43: ClassNames.ROAD.value,  # pothole
                              44: ClassNames.STATIC_STRUCTURE.value,
                              45: ClassNames.STATIC_STRUCTURE.value,
                              46: ClassNames.STATIC_STRUCTURE.value,
                              47: ClassNames.STATIC_STRUCTURE.value,
                              48: ClassNames.STATIC_STRUCTURE.value,
                              49: ClassNames.STATIC_STRUCTURE.value,
                              50: ClassNames.STATIC_STRUCTURE.value,
                              51: ClassNames.STATIC_STRUCTURE.value,
                              52: ClassNames.VEHICLE.value,
                              54: ClassNames.VEHICLE.value,
                              55: ClassNames.VEHICLE.value,
                              56: ClassNames.VEHICLE.value,
                              57: ClassNames.VEHICLE.value,
                              58: ClassNames.VEHICLE.value,
                              59: ClassNames.VEHICLE.value,
                              60: ClassNames.VEHICLE.value,
                              61: ClassNames.VEHICLE.value,
                              62: ClassNames.VEHICLE.value,  # Vehicles
                              53: ignore_label,  # Boat
                              63: ignore_label,
                              64: ignore_label,
                              65: ignore_label}
        # self.class_weights = torch.FloatTensor([0.8373, 1.1, 1.1, 1.2,
        #                                         1.05, 0.9, 0.9, 1.1, 0.8786, 0.7, 1]).cuda()
        self.class_weights = torch.FloatTensor([1, 1.2, 1.2, 1.5,
                                                1.05, 0.9, 0.9, 1.1, 0.9, 0.9, 1]).cuda()

        self.class_thresholds = (100000, 100000, 900000, 100000, 100000, 1000000, 80, 20, 1000000)

    def handle_pred_maps(self, pred):
        assert len(pred.shape) == 4
        assert pred.shape[0] == 1, 'support only batch size 1'
        pred = pred.squeeze()
        res = np.zeros(pred.shape[1:])
        for i, map_ in enumerate(pred):
            map_ = map_.cpu().numpy()
            res[map_ > self.class_thresholds[i]] = i + 1
        return res
    
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root,'cityscapes',item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label = cv2.imread(os.path.join(self.root,'cityscapes',item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        image, label = self.gen_sample(image, label, 
                                self.multi_scale, self.flip)
        
        # label = cv2.resize(label, (960, 720), interpolation=cv2.INTER_NEAREST)
        # image = cv2.resize(image, (960, 720), interpolation=cv2.INTER_LINEAR)

        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]

            preds = F.interpolate(
                preds, (ori_height, ori_width), 
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )            
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
