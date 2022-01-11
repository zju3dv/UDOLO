import torch
import numpy as np
from .kalman_utils import AB3DMOT


class Map(object):
    '''

    '''

    def __init__(self, score_decay, boxes_decay, sem_score_decay, num_class=1):
        MAX_NUM = 3000
        self.object_scores_map = torch.zeros(MAX_NUM)
        self.object_num_map = torch.zeros(MAX_NUM, )
        self.object_semantics_map = torch.zeros(MAX_NUM, num_class)
        self.score_decay = score_decay
        self.boxes_decay = boxes_decay
        self.sem_score_decay = sem_score_decay
        self.device = None
        self.object_dof_map = torch.zeros(MAX_NUM, 7)
        self.kalman_filter = AB3DMOT()
        self.num_class = 21

    def kalman_update(self, boxes, scores, sem_scores):
        '''

        :param boxes: xyzhwl_ry
        :return:
        '''
        boxes = boxes[:, [3, 4, 5, 0, 1, 2, 6]]
        boxes = boxes.data.cpu().numpy()
        scores = scores.data.cpu().numpy()
        sem_scores = sem_scores.data.cpu().numpy()
        additional_info = np.zeros((boxes.shape[0], 28))
        additional_info[:, -1] = scores
        additional_info[:, -22:-1] = sem_scores
        info = {'dets': boxes, 'info': additional_info}
        trackers = self.kalman_filter.update(info)
        updated_boxes = torch.Tensor([d[0:7] for d in trackers]).cuda()
        ids = torch.Tensor([d[7] for d in trackers]).cuda()
        updated_scores = torch.Tensor([d[-1] for d in trackers]).cuda()
        updated_sem_scores = torch.Tensor([d[-22:-1] for d in trackers]).cuda()
        updated_boxes = updated_boxes[:, [3, 4, 5, 0, 1, 2, 6]]
        return updated_boxes, ids, updated_scores, updated_sem_scores

    def kalman_predict(self, ):
        return torch.Tensor(self.kalman_filter.predict()).cuda()[:, [3, 4, 5, 0, 1, 2, 6]]

    def update_ids_scores_semantics(self, ids, scores, sem_scores):
        '''

        :param ids:
        :param scores:
        :return:
        '''
        # ----------------generate new id-----------------------------
        new_ids_ind = torch.nonzero(ids == 0).squeeze(1).long()
        if len(new_ids_ind) != 0:
            all_ids = torch.nonzero(self.object_num_map).squeeze(1)
            if len(all_ids) == 0:
                new_first_id = 1
            else:
                new_first_id = all_ids.max() + 1
            ids[new_ids_ind] = torch.arange(len(new_ids_ind)).cuda() + new_first_id

        # --------------update scores semantics------------------------
        ids = ids.long()
        pre_scores = self.object_scores_map[ids]
        fused_scores = (scores + self.score_decay * pre_scores) / (1 + self.score_decay)
        self.object_scores_map[ids] = fused_scores

        pre_sem_scores = self.object_semantics_map[ids]
        fused_sem_scores = (sem_scores + self.sem_score_decay * pre_sem_scores) / (1 + self.sem_score_decay)
        self.object_semantics_map[ids] = fused_sem_scores

        return ids

    def update_boxes(self, boxes, ids):
        '''

        :param ids:
        :param scores:
        :return:
        '''
        # --------------update boxes------------------------
        ids = ids.long()
        pre_dof = self.object_dof_map[ids]
        num = self.object_num_map[ids].unsqueeze(-1)
        fused_boxes = (boxes + self.boxes_decay * pre_dof) / (1 + self.boxes_decay * num)
        self.object_num_map[ids] += 1
        self.object_dof_map[ids] = fused_boxes
        return fused_boxes, ids, torch.arange(fused_boxes.shape[0]).to(self.device)

    def get_boxes(self, ):
        all_boxes_index = torch.nonzero(self.object_num_map).squeeze(1)
        all_boxes = self.object_dof_map[all_boxes_index]
        return all_boxes, all_boxes_index


class VoxelMapGrid(Map):
    '''

    '''

    def __init__(self, voxel_size, detect_thresh, voxel_area, score_decay, boxes_decay, sem_score_decay, num_class,
                 world_coord):
        super(VoxelMapGrid, self).__init__(score_decay, boxes_decay, sem_score_decay, num_class)
        if world_coord == 'depth':
            self.slice = [0, 1]
        elif world_coord == 'camera':
            self.slice = [0, 2]
        self.voxel_size = torch.from_numpy(np.array(voxel_size)).float()
        self.detect_thresh = detect_thresh
        self.voxel_area = voxel_area
        self.voxel_map = torch.zeros(self.voxel_area)
        self.device = None

    def reset(self, device):
        self.voxel_map = self.voxel_map.zero_().to(device)
        self.object_dof_map = self.object_dof_map.zero_().to(device)
        self.object_scores_map = self.object_scores_map.zero_().to(device)
        self.object_num_map = self.object_num_map.zero_().to(device)
        self.object_semantics_map = self.object_semantics_map.zero_().to(device)
        self.device = device

    def update_map(self, backbone_xyz, pose_camera_to_world=None):
        '''
        It's trading space for time
        :param backbone_xyz:
        :param targets:
        :return: ids for area need detected
        '''

        if pose_camera_to_world is not None:
            ones = torch.ones(backbone_xyz.shape[0], device=self.device).unsqueeze(1)
            world_pts = pose_camera_to_world @ torch.cat([backbone_xyz, ones], dim=1).permute(1, 0)
            world_pts = world_pts[self.slice].permute(1, 0).contiguous()
        else:
            world_pts = backbone_xyz
        discrete_pts = torch.round(world_pts / self.voxel_size.to(self.device)).type(torch.long)
        discrete_pts[:, 0] = discrete_pts[:, 0] + self.voxel_area[0] / 2
        discrete_pts[:, 1] = discrete_pts[:, 1] + self.voxel_area[1] / 2
        valid_mask = ((discrete_pts[:, 0] >= 0) & (
                discrete_pts[:, 0] < self.voxel_area[0])) & ((
                                                                     discrete_pts[:, 1] >= 0) & (
                                                                     discrete_pts[:, 1] < self.voxel_area[1]))
        valid_ind = torch.nonzero(valid_mask).squeeze(1)
        value = self.voxel_map[discrete_pts[valid_ind, 0], discrete_pts[valid_ind, 1]]
        self.voxel_map[discrete_pts[valid_ind, 0], discrete_pts[valid_ind, 1]] += 1

        mask_seen_ind = torch.nonzero(value > self.detect_thresh).squeeze(1)
        valid_mask.zero_()[valid_ind[mask_seen_ind]] = 1
        mask_seen_area = valid_mask
        return mask_seen_area
