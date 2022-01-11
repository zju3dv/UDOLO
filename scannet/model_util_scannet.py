# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import torch


class ScannetSVDatasetConfig(object):
    def __init__(self):
        self.num_class = 21
        self.num_heading_bin = 12
        self.num_size_cluster = 21

        self.type2class = {'chair': 0, 'table': 1, 'cabinet': 2, 'trash bin': 3, 'bookshelf': 4, 'display': 5,
                           'sofa': 6, 'bathtub': 7, 'bed': 8, 'file cabinet': 9, 'bag': 10, 'printer': 11, 'washer': 12,
                           'lamp': 13, 'microwave': 14, 'stove': 15, 'basket': 16, 'bench': 17, 'laptop': 18,
                           'computer keyboard': 19, 'other': 20}

        self.class2type = {self.type2class[t]: t for t in self.type2class}

        self.typelong_mean_size = {}
        with open('scannet/average_scan2cad.txt', 'r') as f:
            for line in f.readlines():
                type_cat, size = line.split(': ')
                size = size[1:-3].split(' ')
                size_ = []
                for j, s in enumerate(size):
                    if len(s) != 0:
                        size_.append(s)
                size = [float(size_[i]) for i in [0, 2, 1]]
                self.typelong_mean_size[type_cat] = size

        self.mean_size_arr = []
        self.type_mean_size = {}
        for i in range(self.num_class):
            type = self.class2type[i]
            for key, value in self.typelong_mean_size.items():
                key = key.split(',')
                if type in key:
                    self.mean_size_arr.append(value)
                    self.type_mean_size[type] = value
                    break
        self.mean_size_arr.append([1, 1, 1])
        self.type_mean_size['other'] = [1, 1, 1]
        self.mean_size_arr = np.array(self.mean_size_arr)

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        mean_size = self.type_mean_size[self.class2type[pred_cls]]
        return mean_size + residual

    def class2size_batch(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        # todo change when multi class
        assert pred_cls.shape[0] == 1
        # mean_size = torch.stack(
        #     [torch.from_numpy(self.type_mean_size[self.class2type[int(cls.data.cpu())]]).cuda()
        #      for
        #      cls in pred_cls[0]]).float().unsqueeze(0)
        mean_size = torch.from_numpy(
            self.type_mean_size[self.class2type[int(pred_cls[0, 0].data.cpu())]]).cuda().float().unsqueeze(0).unsqueeze(
            0)
        # mean_size = self.type_mean_size[self.class2type[pred_cls]]
        return mean_size + residual

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from
            class center angle to current angle.

            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        angle = angle % (2 * np.pi)
        assert (angle >= 0 and angle <= 2 * np.pi)
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def class2angle_batch(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls.float() * angle_per_class
        angle = angle_center + residual
        if to_label_format:
            angle[angle > np.pi] = angle[angle > np.pi] - 2 * np.pi
        return angle

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb
