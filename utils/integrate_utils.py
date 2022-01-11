import torch
import copy
import iou3d_cuda


def boxes3d_to_bev_torch(boxes3d):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :return:
        boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

    cu, cv = boxes3d[:, 0], boxes3d[:, 2]
    half_l, half_w = boxes3d[:, 5] / 2, boxes3d[:, 4] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev


def boxes_iou_bev(boxes_a, boxes_b):
    """
    :param boxes_a: (M, 5)
    :param boxes_b: (N, 5)
    :return:
        ans_iou: (M, N)
    """

    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = boxes3d_to_bev_torch(boxes_b)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_cuda.boxes_overlap_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev)

    # height overlap
    boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, 3]).view(-1, 1)
    boxes_a_height_max = boxes_a[:, 1].view(-1, 1)
    boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, 3]).view(1, -1)
    boxes_b_height_max = boxes_b[:, 1].view(1, -1)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)

    return iou3d


def _pc_bbox3_filter(bbox3, pc):
    """
    Find point cloud inside a bounding box.
    """
    point = copy.deepcopy(pc)
    v45 = bbox3[5] - bbox3[4]
    v40 = bbox3[0] - bbox3[4]
    v47 = bbox3[7] - bbox3[4]
    point -= bbox3[4]
    m0 = torch.matmul(point, v45)
    m1 = torch.matmul(point, v40)
    m2 = torch.matmul(point, v47)

    cs = []
    for m, v in zip([m0, m1, m2], [v45, v40, v47]):
        c0 = 0 < m
        c1 = m < torch.matmul(v, v)
        c = c0 & c1
        cs.append(c)
    cs = cs[0] & cs[1] & cs[2]
    num_passed = torch.sum(cs)
    passed_inds = torch.nonzero(cs).squeeze(1)
    return num_passed, passed_inds, cs


def _pc_bbox3_filter_batch(bbox3, pc):
    '''
    Find point cloud inside a bounding box.
    :param bbox3:
    :param pc:
    :return:
    '''
    point = copy.deepcopy(pc).unsqueeze(0)
    v45 = bbox3[:, 5] - bbox3[:, 4]
    v40 = bbox3[:, 0] - bbox3[:, 4]
    v47 = bbox3[:, 7] - bbox3[:, 4]
    point = point - bbox3[:, 4:5]
    m0 = torch.matmul(point, v45.unsqueeze(-1))
    m1 = torch.matmul(point, v40.unsqueeze(-1))
    m2 = torch.matmul(point, v47.unsqueeze(-1))
    cs = []
    for m, v in zip([m0, m1, m2], [v45, v40, v47]):
        c0 = 0 < m
        c1 = m < v.unsqueeze(1) @ v.unsqueeze(-1)
        c = c0 & c1
        cs.append(c)
    cs = cs[0] & cs[1] & cs[2]
    cs = cs.squeeze(-1)
    return cs


def _pc_bbox3_filter_bev(bbox3, pc, coord='camera'):
    '''
    Find point cloud inside a bounding box in bird view.
    :param bbox3:
    :param pc:
    :return:
    '''
    if coord == 'camera':
        point = pc[:, [0, 2]].unsqueeze(0)
    else:
        point = pc[:, [0, 1]].unsqueeze(0)
    bbox3 = bbox3[:, :, [0, 2]]
    v40 = bbox3[:, 0] - bbox3[:, 4]
    v47 = bbox3[:, 7] - bbox3[:, 4]
    point = point - bbox3[:, 4:5]
    m0 = torch.matmul(point, v40.unsqueeze(-1))
    m1 = torch.matmul(point, v47.unsqueeze(-1))
    cs = []
    for m, v in zip([m0, m1], [v40, v47]):
        c0 = 0 < m
        c1 = m < v.unsqueeze(1) @ v.unsqueeze(-1)
        c = c0 & c1
        cs.append(c)
    cs = cs[0] & cs[1]
    cs = cs.squeeze(-1)
    return cs
