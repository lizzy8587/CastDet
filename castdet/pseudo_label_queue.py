#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   pseudo_label_queue.py
@Version      :   1.0
@Time         :   2024/04/08 21:13:11
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   Implementation of `pseudo label queue`.
'''

import torch
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
import random
import numpy as np
from torchvision.transforms import Compose
from torchvision.transforms.functional import to_pil_image, to_tensor, normalize
import imagesize
import os
import imagesize
import cv2
from mmdet.datasets.transforms import Resize as mmResize
import atexit


def transform_with_bboxes(input):
    mean=(0.48145466, 0.4578275, 0.40821073)
    std=(0.26862954, 0.26130258, 0.27577711)
    img = to_pil_image(cv2.cvtColor(input['img'], cv2.COLOR_BGR2RGB)).convert('RGB')
    img = to_tensor(img)
    input['img'] = normalize(img, mean, std)

    return input

coco_transforms_2 = Compose([
    mmResize(scale=(1333, 800), keep_ratio=True),
    transform_with_bboxes,
])

default_transforms = Compose([
    # mmResize(width=800, height=800, keep_ratio=True),
    transform_with_bboxes,
])

custom_transforms = {
    'default_transforms': default_transforms,
    'nwpu_transforms': default_transforms,
    'coco_transforms': coco_transforms_2
}

@MODELS.register_module()
class PseudoQueue:
    def __init__(
        self, unseen_ids: tuple,
        batch_num: int=4,
        start_train_num: int=10,
        sample_prob: tuple=None,
        transform: str='default_transforms',
        **kwargs
    ):
        self.unseen_ids = unseen_ids
        self.batch_num = batch_num
        self.transform = custom_transforms[transform]
        self.start_train_num = start_train_num
        self.unseen_imgs = dict(zip(unseen_ids, [[] for i in unseen_ids]))
        self.name2detSample = {}
        self.sample_prob = sample_prob if sample_prob is not None else tuple([1/len(unseen_ids)] * len(unseen_ids))

        self.start_train_iter = kwargs.get('start_train_iter', 0)
        self.cur_iter = 0
        self.init_imgs = []
        self.save_path = kwargs.get('save_path', None)
        self.class_sampling = kwargs.get('class_sampling', True)
        if kwargs.get('initialize', False):
            self.initialize(**kwargs)
        atexit.register(self.save_queue)
        

    def save_queue(self):
        if self.save_path is not None:
            save_path = os.path.join(os.path.dirname(self.save_path), f"{self.cur_iter}_" + os.path.basename(self.save_path))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data = {
                'cur_iter': self.cur_iter,
                'name2detSample': self.name2detSample,
                'unseen_imgs': self.unseen_imgs
            }
            np.savez(save_path, **data)
            print(f'Successfully save queue in {save_path}')
            
    def initialize(self, **kwargs):
        if self.save_path is not None and os.path.exists(self.save_path):
            data = np.load(self.save_path, allow_pickle=True)
            data = dict(data)
            self.cur_iter = data['cur_iter'].item()
            self.name2detSample = data['name2detSample'].item()
            self.unseen_imgs = data['unseen_imgs'].item()
            print(f'Successfully load info from {self.save_path}')
        elif kwargs.get('init_imgs_path', False):
            with open(kwargs['init_imgs_path'], 'r') as fp:
                data = fp.readlines()
                data = [i.strip('\n') for i in data]
            self.init_imgs = data

    def start_training(self):
        if self.class_sampling:
            for v in self.unseen_imgs.values():
                if len(v) < self.start_train_num:
                    return False
            return True
        else:
            return len(self.name2detSample) >= self.start_train_num
    
    def start_update(self):
        if self.cur_iter % 2000 == 0:
            self.save_queue()
        self.cur_iter += 1
        return self.cur_iter >= self.start_train_iter
        
    def update_pseudo_queue(self, img_path, bboxes, labels, scores=None):
        """update the pseudo queue"""
        to_update = False
        img_name = img_path.split('/')[-1]

        # remove and update the unseen queue
        for idx in self.unseen_ids:
            if img_name in self.unseen_imgs[idx]:
                self.unseen_imgs[idx].remove(img_name)
        
        for idx in set(labels.tolist()):
            if idx in self.unseen_ids:
                self.unseen_imgs[idx].append(img_name)
                to_update = True
        
        # update the instaces queue
        if to_update:
            self.name2detSample[img_name] = {
                "img_path": img_path,
                "bboxes": bboxes,
                "labels": labels,
                "scores": scores
            }

    def get_image_detsample(self):
        """randomly return pseudo samples from the queue"""
        imgs, detsample_list = [], []
        if not self.start_training():
            return imgs, detsample_list

        if self.class_sampling:
            categories = np.random.choice(a=self.unseen_ids, size=self.batch_num, p=self.sample_prob)
            img_list = [random.choice(self.unseen_imgs[c]) for c in categories]
        else:
            img_list = [random.choice(list(self.name2detSample.keys())) for i in range(self.batch_num)]

        samples, results = [], []
        for img_name in img_list:
            sample = self.name2detSample[img_name]
            result = self.transform({
                'img': cv2.imread(sample['img_path']),
                'gt_bboxes': sample['bboxes']
            })
            imgs.append(result['img'].unsqueeze(0))
            samples.append(sample)
            results.append(result)

        sizes = [(img.shape[-2], img.shape[-1]) for img in imgs]
        max_h = max(s[0] for s in sizes)
        max_w = max(s[1] for s in sizes)

        for img_name, sample, result in zip(img_list, samples, results):
            gt_detSample = self.bbox2detSample(img_path=sample['img_path'],
                                               bboxes=result['gt_bboxes'],
                                               scores=sample['scores'],
                                               labels=sample['labels'],
                                               batch_input_shape=(max_h, max_w),
                                               pad_shape=(max_h, max_w),
                                               **result)
            
            detsample_list.append(gt_detSample)
        
        imgs = [torch.nn.functional.pad(img, (0, max_w - img.size(-1), 0, max_h - img.size(-2))) for img in imgs]

        imgs = torch.cat(imgs, dim=0).cuda()
        return imgs, detsample_list

    def bbox2detSample(self, img_path, bboxes=None, labels=None, scores=None, **kwargs):
        """return instance structrue"""
        w, h =imagesize.get(img_path)

        data_sample = DetDataSample(metainfo={
            'img_path': img_path,
            'scale_factor': kwargs.get('scale_factor', (1.0, 1.0)),
            'img_shape': kwargs.get('img_shape', (h, w)),
            'pad_shape': kwargs.get('pad_shape', (h, w)),
            'batch_input_shape': kwargs.get('batch_input_shape', (h, w)),
            'ori_shape': (h, w),
            'keep_ratio': kwargs.get('keep_ratio', True),
            'homography_matrix': kwargs.get('homography_matrix', None)
        })
        gt_instances = InstanceData()
        if bboxes is not None:
            gt_instances.bboxes = torch.tensor(bboxes).cuda()
        if labels is not None:
            gt_instances.labels = torch.tensor(labels).cuda()
        if scores is not None:
            gt_instances.scores = torch.tensor(scores).cuda()
        data_sample.gt_instances = gt_instances
        
        return data_sample
