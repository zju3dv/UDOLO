import torch
import numpy as np
from utils.integrate_utils import _pc_bbox3_filter, _pc_bbox3_filter_batch, _pc_bbox3_filter_bev, boxes_iou3d_gpu
from utils.bounding_box_3d import Box3DList
from utils.voxel_map import VoxelMapGrid


class Integrate(object):
    '''

    '''

    def __init__(self, voxel_size, detect_thresh, map_area, score_decay, boxes_decay, sem_score_decay,
                 valid_score,
                 num_proposal_per_box,
                 iou_thresh, view_coord, world_coord,
                 num_class=1,
                 top_n=None,
                 top_n_pred=None,
                 use_kalman=False,
                 use_backend_conv=False,
                 aug_score_based=False,
                 num_point_thresh=10,
                 track_area_extra_width=0.2,
                 ):
        '''

        :param voxel_size:
        :param detect_thresh:
        :param map_area:
        :param score_decay:
        :param boxes_decay:
        :param valid_score:
        :param num_proposal_per_box:
        :param iou_thresh:
        :param view_coord:  'camera' : X-right,Y-down,Z-forward ; 'depth' : Flip X-right,Y-forward,Z-up
        :param world_coord:
        :param use_backend_conv:
        :param aug_score_based:
        :param num_point_thresh:
        :param track_area_extra_width:
        :param dataset:
        '''
        self.map = VoxelMapGrid(voxel_size, detect_thresh, map_area,
                                score_decay, boxes_decay, sem_score_decay, num_class, world_coord)
        self.frame_id = 0
        self.pose = []

        self.match_tra_box3d = None
        self.match_pre_ids = None
        self.tra_box3d = None
        self.pre_ids = None
        self.track_status = None
        self.rpn_post_nms_top_n_integrate = None
        self.unseen_area_ind_in_valid_area = None
        self.pose_camera_to_world = None

        self.track_area_extra_width = track_area_extra_width
        self.valid_score = valid_score
        self.num_point_thresh = num_point_thresh
        self.aug_score_based = aug_score_based
        self.use_backend_cov = use_backend_conv
        self.num_proposal_per_box = num_proposal_per_box
        self.iou_thresh = iou_thresh
        self.view_coord = view_coord
        self.world_coord = world_coord
        self.view_frame = 'rect' if view_coord == 'camera' else 'velodyne'
        self.world_frame = 'rect' if world_coord == 'camera' else 'velodyne'
        self.top_n = top_n
        self.top_n_pred = top_n_pred
        self.use_kalman = use_kalman

    # ------------------------------------INIT------------------------------------------------------
    def init(self, frame_id, pose_camera_to_world):
        self.frame_id = frame_id
        self.pose_camera_to_world = pose_camera_to_world

    # -------------------------------------GET---------------------------------------
    def get_match_box3d(self, mode):
        if self.tra_box3d is not None:
            box3d = Box3DList(self.match_tra_box3d, mode='xyzhwl_ry', frame=self.view_frame).convert(mode).bbox_3d
            if mode == 'corners':
                box3d = box3d.view(-1, 8, 3)
            scores = self.map.object_scores_map[self.match_pre_ids]
            semantics = self.map.object_semantics_map[self.match_pre_ids]
            return box3d, self.match_pre_ids, scores, semantics
        else:
            return None, None, None, None

    # --------------------------------EARLY-INTEGRATE------------------------------------------------------
    def process_tracklet(self, tra_box3d, pre_ids, backbone_xyz):
        '''
        filter boxes
        :param tra_box3d:
        :param pre_ids:
        :param backbone_xyz:
        :return:
        '''
        mask_track_area = _pc_bbox3_filter_batch(tra_box3d, backbone_xyz)
        mask_valid = mask_track_area.sum(dim=1) > self.num_point_thresh
        match_fov_ind = torch.nonzero(mask_valid).squeeze(
            1).data.cpu().numpy()

        if not self.use_kalman:
            scores_map = self.map.object_scores_map
            scores = scores_map[pre_ids]
        else:
            scores = torch.full((pre_ids.shape[0],), 100.0).cuda()
        invalid_score_mask = scores < self.valid_score
        mask_valid &= (invalid_score_mask == False)
        fov_ind = torch.nonzero(mask_valid).squeeze(
            1).data.cpu().numpy()

        invalid_score_track_ind = torch.nonzero(mask_track_area[invalid_score_mask])
        invalid_score_ind = torch.nonzero(invalid_score_mask).squeeze(1)
        if len(invalid_score_track_ind) != 0:
            mask_track_area = mask_track_area.int()
            mask_track_area[invalid_score_ind[invalid_score_track_ind[:, 0]], invalid_score_track_ind[:, 1]] = -99
        mask_track_area_sum = mask_track_area.sum(dim=0) > 0
        mask_track_area_unvalid = mask_track_area.sum(dim=0) < 0
        mask_track_area_sum = mask_track_area_sum.int()
        mask_track_area_sum[mask_track_area_unvalid] = -1
        return match_fov_ind, fov_ind, mask_track_area_sum

    def aug_box3d(self, tra_box3d, pre_ids, proposal_per_box):
        '''

        :param tra_box3d:
        :param pre_ids:
        :param proposal_per_box:
        :return:
        '''
        torch.manual_seed(123)
        tra_box3d = tra_box3d.unsqueeze(1).expand(-1, proposal_per_box, -1)
        pre_ids = pre_ids.unsqueeze(1).expand(-1, proposal_per_box).contiguous()
        norm_scores = None
        if self.aug_score_based:
            scores = self.map.object_scores_map[pre_ids[:, 0]]
            norm_scores = torch.sigmoid(scores)
            norm_scores = (1 / norm_scores)
            norm_scores = norm_scores.unsqueeze(-1).unsqueeze(-1)
        std_config = [0.25, 0.15, 0.25, 0.02, 0.0125, 0.0125, np.pi / 48]
        # xyzhwl_ry
        num_pre_box = tra_box3d.shape[0]
        if norm_scores is not None:
            box3d = tra_box3d + torch.randn(num_pre_box, proposal_per_box, 7).cuda() * (torch.Tensor(
                std_config).cuda().unsqueeze(0).unsqueeze(0) * norm_scores)
        else:
            box3d = tra_box3d + torch.randn(num_pre_box, proposal_per_box, 7).cuda() * torch.Tensor(
                std_config).cuda().unsqueeze(0).unsqueeze(0)
        return box3d, pre_ids

    def get_tracklet_from_backend(self, backbone_xyz):
        '''

        :param backbone_xyz:
        :return: box3d distribution sampled, ids, number of output in detection area

        '''
        if self.use_kalman:
            tra_box3d = self.map.kalman_predict()
            pre_ids = torch.zeros(tra_box3d.shape[0]).cuda().long()
            tra_box3d = Box3DList(tra_box3d, mode='xyzhwl_ry').convert('ry_lhwxyz').bbox_3d.view(-1, 7)
        else:
            tra_box3d, pre_ids = self.map.get_boxes()
        # convert 3d bounding box from world coordinate to camera coordinate
        tra_box3d = Box3DList(tra_box3d, mode='ry_lhwxyz', frame=self.world_frame).convert(
            'corners').bbox_3d.view(-1, 3)
        pose_world_to_camera = torch.inverse(self.pose_camera_to_world.cpu()).cuda()
        ones = torch.ones(tra_box3d.shape[0]).cuda().unsqueeze(1)
        tra_box3d = pose_world_to_camera @ torch.cat([tra_box3d, ones], dim=1).permute(1, 0)
        tra_box3d = tra_box3d[:3].permute(1, 0).contiguous()
        tra_box3d = tra_box3d.view(-1, 8, 3).contiguous()

        # filter boxes
        match_fov_ind, fov_ind, mask_track_area = self.process_tracklet(tra_box3d, pre_ids, backbone_xyz)
        tra_box3d = Box3DList(tra_box3d, mode='corners', frame=self.view_frame).convert('xyzhwl_ry').bbox_3d
        if len(match_fov_ind) != 0:
            self.match_tra_box3d, self.match_pre_ids = tra_box3d[match_fov_ind], pre_ids[match_fov_ind]
        else:
            self.match_tra_box3d, self.match_pre_ids = None, None
        if len(fov_ind) != 0:
            tra_box3d = tra_box3d[fov_ind]
            pre_ids = pre_ids[fov_ind]
            num_tra = tra_box3d.shape[0]
            if self.top_n is not None:
                proposal_per_box = (
                                           self.top_n - self.top_n_pred) // num_tra
                rpn_post_nms_top_n_integrate = self.top_n - proposal_per_box * num_tra
                if proposal_per_box == 0:
                    proposal_per_box = 1
            else:
                proposal_per_box = self.num_proposal_per_box
            tra_box3d, pre_ids = self.aug_box3d(tra_box3d, pre_ids, proposal_per_box)

            self.tra_box3d = tra_box3d.view(-1, 7)
            self.pre_ids = pre_ids.view(-1)

        else:
            # No tracklet in front of view
            self.tra_box3d = None
            self.pre_ids = None
        return mask_track_area

    def early(self, pts):
        '''

        :param pts:
        :return:

        '''

        if self.frame_id == 0:
            self.match_tra_box3d = self.tra_box3d = self.match_pre_ids = self.pre_ids = None
            self.map.reset(pts.device)
        mask_seen_area = self.map.update_map(pts, pose_camera_to_world=self.pose_camera_to_world)
        unseen_area_ind = torch.nonzero(mask_seen_area == False).squeeze(1)
        self.unseen_area_ind_in_valid_area = torch.arange(len(unseen_area_ind))
        valid_area_ind = unseen_area_ind
        mask_track_area = None
        if self.frame_id != 0:
            # --------------------------------Process n frame(n > 1)--------------------------------------
            mask_track_area = self.get_tracklet_from_backend(pts)
            if mask_track_area is not None:
                valid_area_ind = torch.nonzero((mask_seen_area == False) | (mask_track_area != 0)).squeeze(1)
                seen_area_mask_in_valid_area = mask_seen_area[valid_area_ind]
                track_area_mask_in_valid_area = mask_track_area[valid_area_ind]
                self.unseen_area_ind_in_valid_area = torch.nonzero(
                        (seen_area_mask_in_valid_area == False) | (track_area_mask_in_valid_area < 0)).squeeze(1)

        return valid_area_ind

    # --------------------------------------Matching----------------------------------------------------
    def match_rois(self, rois):
        '''

        :param pre_ids:
        :param rois:
        :return:
        '''
        ids = torch.zeros(rois.shape[0]).cuda().long()
        rois = rois.squeeze(0)
        if self.tra_box3d is not None:
            iou3d = boxes_iou3d_gpu(self.match_tra_box3d, rois)
            max_value, max_ind = iou3d.max(dim=0)
            self.max_value = max_value
            tracked_ind = torch.nonzero(max_value > self.iou_thresh).squeeze(1)
            ids[tracked_ind] = self.match_pre_ids[max_ind[tracked_ind]]
        return ids

    def matching(self, pre_box3d, mode):
        '''

        :param backbone_xyz:
        :param rpn_reg:
        :param rpn_scores_raw:
        :param targets:
        :return:
        '''

        if self.frame_id and self.tra_box3d is not None:
            # --------------------------------Process n frame(n > 1)--------------------------------------
            pre_box3d = Box3DList(pre_box3d, mode=mode, frame=self.view_frame).convert('xyzhwl_ry').bbox_3d
            ids = self.match_rois(pre_box3d)
        else:
            # --------------------------------Process the first frame-------------------------------------
            ids = torch.zeros(pre_box3d.shape[0]).cuda().long()

        return ids

    # ------------------------------------------------LATE_INTEGRATE----------------------------------------------------------
    def late(self, pred, scores, ids, sem_scores, mode):
        if self.use_kalman:
            # convert 3d bounding box from camera coordinate to world coordinate
            pred = Box3DList(pred, mode=mode, frame=self.view_frame).convert("corners")
            pred_box3d = pred.bbox_3d.view(-1, 3)
            ones = torch.ones(pred_box3d.shape[0]).cuda().unsqueeze(1)
            box3d = self.pose_camera_to_world @ torch.cat([pred_box3d, ones], dim=1).permute(1, 0)
            box3d = box3d[:3].permute(1, 0).contiguous().view(-1, 8, 3)
            box3d = Box3DList(box3d, mode='corners', frame=self.world_frame)
            box3d_final = box3d.convert('ry_lhwxyz').bbox_3d

            pred_backend, ids_backend, scores, sem_scores = self.map.kalman_update(box3d_final, scores, sem_scores)

            # convert 3d bounding box from world coordinate to camera coordinate
            box3d_backend = Box3DList(pred_backend, mode='ry_lhwxyz', frame=self.world_frame).convert(
                'corners')
            box3d_backend_final = box3d_backend.bbox_3d.view(-1, 3)

            pose_world_to_camera = torch.inverse(self.pose_camera_to_world.cpu()).cuda()
            ones = torch.ones(box3d_backend_final.shape[0]).cuda().unsqueeze(1)
            box3d_backend_final = pose_world_to_camera @ torch.cat([box3d_backend_final, ones], dim=1).permute(1, 0)
            box3d_backend_final = box3d_backend_final[:3].permute(1, 0).contiguous()
            box3d_backend_final = box3d_backend_final.view(-1, 8, 3).contiguous()
            pred_backend = Box3DList(box3d_backend_final, mode='corners', frame=self.view_frame).convert(mode)
            assert mode == 'corners'
            pred_backend = pred_backend.bbox_3d.view(-1, 8, 3)
            _ = self.map.update_ids_scores_semantics(ids_backend, scores, sem_scores)
        else:
            ids_final = self.map.update_ids_scores_semantics(ids, scores, sem_scores)
            pred_backend, ids_backend = self.connect_with_backend(pred, ids_final, mode)
        return pred_backend, scores, ids_backend, sem_scores

    def connect_with_backend(self, pred_box3d, track_id, mode):
        '''

        :param pred_box3d:
        :param track_id:
        :return:
        '''
        # convert 3d bounding box from camera coordinate to
        # world coordinate (B5-Chairs Dataset) or velodyne coordinate(KITTI Dataset)
        pred = Box3DList(pred_box3d, mode=mode, frame=self.view_frame).convert("corners")
        pred_box3d = pred.bbox_3d.view(-1, 3)
        ones = torch.ones(pred_box3d.shape[0]).cuda().unsqueeze(1)
        box3d = self.pose_camera_to_world @ torch.cat([pred_box3d, ones], dim=1).permute(1, 0)
        box3d = box3d[:3].permute(1, 0).contiguous().view(-1, 8, 3)

        box3d = Box3DList(box3d, mode='corners', frame=self.world_frame)
        box3d_final = box3d.convert('ry_lhwxyz').bbox_3d

        box3d_backend, box3d_backend_ids, keep_id_list = self.map.update_boxes(box3d_final, track_id)

        # convert 3d bounding box from world coordinate (B5-Chairs Dataset) or velodyne coordinate(KITTI Dataset)
        # to camera coordinate
        box3d_backend = Box3DList(box3d_backend, mode='ry_lhwxyz', frame=self.world_frame).convert('corners')
        box3d_backend_final = box3d_backend.bbox_3d.view(-1, 3)

        pose_world_to_camera = torch.inverse(self.pose_camera_to_world.cpu()).cuda()
        ones = torch.ones(box3d_backend_final.shape[0]).cuda().unsqueeze(1)
        box3d_backend_final = pose_world_to_camera @ torch.cat([box3d_backend_final, ones], dim=1).permute(1, 0)
        box3d_backend_final = box3d_backend_final[:3].permute(1, 0).contiguous()
        box3d_backend_final = box3d_backend_final.view(-1, 8, 3).contiguous()
        box3d_backend_final = Box3DList(box3d_backend_final, mode='corners', frame=self.view_frame).convert(mode)

        return box3d_backend_final.bbox_3d.view(-1, 8, 3), box3d_backend_ids
