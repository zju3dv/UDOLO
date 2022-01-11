# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import os
import sys
import numpy as np
from torch.utils.data import Dataset
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from scannet.scannet_utils import get_point_cloud, rotate_view_to_align_box3d, corners2xyzlhw, \
    get_point_votes, rotz, my_compute_box_3d

MAX_NUM_OBJ = 64  # maximum number of objects allowed per scene
import pc_util

from scannet.model_util_scannet import ScannetSVDatasetConfig

DC = ScannetSVDatasetConfig()


class ScannetSVDetectionDataset(Dataset):

    def __init__(self, split_set='train', num_points=20000,
                 use_color=False, use_height=False, augment=False, fix_seed=False):

        self.data_path = os.path.join(BASE_DIR, 'scans')
        assert os.path.exists(self.data_path)
        self.root = self.data_path
        self.split = split_set
        self.item_list = self.get_roidb()

        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment
        self.fix_seed = fix_seed

    def get_roidb(self):
        roidb_filename = os.path.join(BASE_DIR, "oriented_boxes_annotation_train.pkl") if "train" in self.split \
            else os.path.join(BASE_DIR, "oriented_boxes_annotation_val.pkl")

        if os.path.exists(roidb_filename):
            item_list = pickle.load(open(roidb_filename, "rb"))
            print("roidb loaded from {}, totally {} samples".format(roidb_filename, len(item_list)))
            return item_list
        else:
            raise NotADirectoryError

    def type2class(self, types):
        classes = []
        for type in types:
            all_type = type.split(',')
            other = True
            cls = None
            for at in all_type:
                try:
                    cls = DC.type2class[at]
                    other = False
                except:
                    pass
            if other:
                cls = 20
            classes.append(cls)
        return classes

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            angle_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            angle_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            point_votes: (N,3) with votes XYZ
            point_votes_mask: (N,) with 0/1 with 1 indicating the point is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            pcl_color: unused
        """
        # x,y,z -> x,z,-y (cv->)
        rot_mat = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        item = self.item_list[idx]

        scene_name = item["scene_name"]
        image_index = item["im_index"]
        type = item['labels']
        label = self.type2class(type)
        point_cloud = get_point_cloud(self.root, scene_name, image_index)
        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True,
                                                       fix_seed=self.fix_seed)

        camera_anno_box3d = item["camera_anno_box3d"]
        camera_anno_box3d = [_[[1, 0, 4, 5, 2, 3, 7, 6]] for _ in camera_anno_box3d]

        # conversion 1 (pc and box): camera coordinates -> aligned coords (xz plane are parallel to the ground)
        camera_anno_box3d, rotation_matrix = rotate_view_to_align_box3d(item["Tr_camera_to_scan"], camera_anno_box3d)
        point_cloud[:, :3] = np.matmul(rotation_matrix, point_cloud[:, :3].T).T

        bboxes = corners2xyzlhw(camera_anno_box3d)

        # conversion 2 (pc and box): x,y,z -> x,z,-y (cv->)
        bboxes[:, :3] = np.matmul(rot_mat, bboxes[:, :3].T).T
        bboxes[:, 3:6] = bboxes[:, 3:6][:, [0, 2, 1]]  # xyzlwh_ry, ry is already -ry, coordinate convert: h <=> w
        point_cloud[:, :3] = np.matmul(rot_mat, point_cloud[:, :3].T).T

        rotation_matrix4x4 = np.eye(4)
        rotation_matrix4x4[:3, :3] = rotation_matrix
        pose_camera_to_scan = item['Tr_camera_to_scan'] @ np.linalg.inv(rotation_matrix4x4)

        if 'val' in self.split:
            # don't need votes
            point_votes = np.zeros((self.num_points, 10))
        else:
            point_votes = get_point_votes(point_cloud, bboxes)

        bboxes = np.concatenate([bboxes, np.array(label)[..., np.newaxis]], axis=1)
        bboxes[:, 3:6] = bboxes[:, 3:6] / 2
        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                bboxes[:, 0] = -1 * bboxes[:, 0]
                bboxes[:, 6] = np.pi - bboxes[:, 6]
                point_votes[:, [1, 4, 7]] = -1 * point_votes[:, [1, 4, 7]]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
            rot_mat = rotz(rot_angle)

            point_votes_end = np.zeros_like(point_votes)
            point_votes_end[:, 1:4] = np.dot(point_cloud[:, 0:3] + point_votes[:, 1:4], np.transpose(rot_mat))
            point_votes_end[:, 4:7] = np.dot(point_cloud[:, 0:3] + point_votes[:, 4:7], np.transpose(rot_mat))
            point_votes_end[:, 7:10] = np.dot(point_cloud[:, 0:3] + point_votes[:, 7:10], np.transpose(rot_mat))

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 0:3] = np.dot(bboxes[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 6] -= rot_angle
            point_votes[:, 1:4] = point_votes_end[:, 1:4] - point_cloud[:, 0:3]
            point_votes[:, 4:7] = point_votes_end[:, 4:7] - point_cloud[:, 0:3]
            point_votes[:, 7:10] = point_votes_end[:, 7:10] - point_cloud[:, 0:3]

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random() * 0.3 + 0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            point_cloud[:, 0:3] *= scale_ratio
            bboxes[:, 0:3] *= scale_ratio
            bboxes[:, 3:6] *= scale_ratio
            point_votes[:, 1:4] *= scale_ratio
            point_votes[:, 4:7] *= scale_ratio
            point_votes[:, 7:10] *= scale_ratio
            if self.use_height:
                point_cloud[:, -1] *= scale_ratio[0, 0]

        # ------------------------------- LABELS ------------------------------
        box3d_centers = np.zeros((MAX_NUM_OBJ, 3))
        box3d_sizes = np.zeros((MAX_NUM_OBJ, 3))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        label_mask = np.zeros((MAX_NUM_OBJ))
        label_mask[0:bboxes.shape[0]] = 1
        max_bboxes = np.zeros((MAX_NUM_OBJ, 8))
        max_bboxes[0:bboxes.shape[0], :] = bboxes

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            box3d_center = bbox[0:3]
            angle_class, angle_residual = DC.angle2class(bbox[6])
            # NOTE: The mean size stored in size2class is of full length of box edges,
            # while in sunrgbd_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here
            box3d_size = bbox[3:6] * 2
            size_class, size_residual = DC.size2class(box3d_size, DC.class2type[semantic_class])
            box3d_centers[i, :] = box3d_center
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            size_residuals[i] = size_residual
            box3d_sizes[i, :] = box3d_size

        target_bboxes_mask = label_mask
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            corners_3d = my_compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])
            # compute axis aligned box
            xmin = np.min(corners_3d[:, 0])
            ymin = np.min(corners_3d[:, 1])
            zmin = np.min(corners_3d[:, 2])
            xmax = np.max(corners_3d[:, 0])
            ymax = np.max(corners_3d[:, 1])
            zmax = np.max(corners_3d[:, 2])
            target_bbox = np.array(
                [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2, xmax - xmin, ymax - ymin, zmax - zmin])
            target_bboxes[i, :] = target_bbox

        point_votes_mask = point_votes[:, 0]
        point_votes = point_votes[:, 1:]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:, 0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:bboxes.shape[0]] = bboxes[:, -1]  # from 0 to 9
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['max_gt_bboxes'] = max_bboxes

        ret_dict['scene_name'] = scene_name

        if 'train' not in self.split:
            point_cloud_camera = (np.linalg.inv(rot_mat) @ point_cloud[:, :3].T).T
            ret_dict['point_clouds_camera'] = point_cloud_camera
            ret_dict['pose_camera_to_scan'] = pose_camera_to_scan
            ret_dict['frame_id'] = item['frame_index']

        return ret_dict


if __name__ == '__main__':
    dset = ScannetSVDetectionDataset(split_set='val', use_height=True, num_points=16384)
    for i_example in range(9000):
        # try:
        example = dset.__getitem__(i_example)
        # except:
        #     pass
        # pc_util.write_ply(example['point_clouds'], 'pc_{}.ply'.format(i_example))
        # viz_votes(example['point_clouds'], example['vote_label'],
        #           example['vote_label_mask'], name=i_example)
        # viz_obb(pc=example['point_clouds'], label=example['center_label'],
        #         mask=example['box_label_mask'],
        #         angle_classes=None, angle_residuals=None,
        #         size_classes=example['size_class_label'], size_residuals=example['size_residual_label'],
        #         name=i_example)
