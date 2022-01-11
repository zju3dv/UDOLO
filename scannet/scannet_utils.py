import os
import numpy as np

import cv2
import transforms3d


def get_matrix(matrix_path):
    with open(matrix_path, 'r') as f:
        lines = f.readlines()
        matrix_lines = []
        for l in lines:
            matrix_lines.append(list(map(lambda x: float(x), l.strip().split(' '))))
        ret = np.array(matrix_lines)
    return ret


def get_point_cloud(root_path, scene_name, image_index):
    image = image_index + ".jpg"

    depth_dir = os.path.join(root_path, scene_name, 'depth')
    intrinsic_dir = os.path.join(root_path, scene_name, 'intrinsic')
    d = cv2.imread(os.path.join(depth_dir, image).replace("jpg", "png"), -1).astype(np.float32)
    d /= 1000
    h, w = d.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xx = xx * d
    yy = yy * d
    pc = np.stack([xx, yy, d, np.ones_like(xx)], axis=2)
    pc = pc.reshape(-1, 4)
    camera_to_screen = get_matrix(os.path.join(intrinsic_dir, "intrinsic_depth.txt"))
    extrinsic_depth = get_matrix(os.path.join(intrinsic_dir, "extrinsic_depth.txt"))
    assert np.sum(extrinsic_depth != np.eye(4)) == 0, \
        "{}'s extrinsic depth is not identity: {}".format(scene_name, extrinsic_depth)
    screen_to_camera = np.linalg.inv(camera_to_screen)
    pc = np.dot(screen_to_camera, pc.T).T
    pos_z = np.nonzero(pc[:, 2] > 0)[0]
    point_cloud = pc[pos_z]
    return point_cloud[:, :3]


def rotate_view_to_align_box3d(Tr_camera_to_scan, box3d_list):
    # world space normal [0, 0, 1]  camera space normal [0, -1, 0]
    z_c = np.dot(np.linalg.inv(Tr_camera_to_scan), np.array([0, 0, 1, 0]))[: 3]
    axis = np.cross(z_c, np.array([0, -1, 0]))
    axis = axis / np.linalg.norm(axis)
    theta = np.arccos(-z_c[1] / (np.linalg.norm(z_c)))
    quat = transforms3d.quaternions.axangle2quat(axis, theta)
    rotation_matrix = transforms3d.quaternions.quat2mat(quat)
    ret_box = []
    for box3d in box3d_list:
        ret_box.append(rotation_matrix.dot(np.array(box3d).T).T)
    return ret_box, rotation_matrix


def corners2xyzlhw(corners_list, coordinate='camera'):
    corners = np.array([_.reshape(-1) for _ in corners_list])
    corners = [corners[:, i * 3: (i + 1) * 3] for i in range(8)]

    # TODO calculate the mean
    dif = (corners[3] - corners[0])
    if coordinate == 'camera':
        ry = -(np.arctan2(dif[:, 2], dif[:, 0])).reshape(-1, 1)
    else:
        ry = np.arctan2(dif[:, 1], dif[:, 0]).reshape(-1, 1)
    xyz = ((corners[6] + corners[0]) / 2).reshape(-1, 3)
    l = np.linalg.norm((corners[0] - corners[3]), axis=1).reshape(-1, 1)
    h = np.linalg.norm((corners[0] - corners[1]), axis=1).reshape(-1, 1)
    w = np.linalg.norm((corners[0] - corners[4]), axis=1).reshape(-1, 1)
    bbox_3d = np.concatenate((xyz, l, h, w, ry), axis=-1)

    return bbox_3d


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def my_compute_box_3d(center, size, heading_angle):
    R = rotz(-1 * heading_angle)
    l, w, h = size
    x_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [w, w, -w, -w, w, w, -w, -w]
    z_corners = [h, h, h, h, -h, -h, -h, -h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    return np.transpose(corners_3d)


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def get_point_votes(pc, boxes):
    N = pc.shape[0]
    point_votes = np.zeros((N, 10))  # 3 votes and 1 vote mask
    point_vote_idx = np.zeros((N)).astype(np.int32)  # in the range of [0,2]
    indices = np.arange(N)
    for obj in boxes:
        # Find all points in this object's OBB
        centroid = obj[:3]
        l = obj[3]
        w = obj[4]
        h = obj[5]
        heading_angle = obj[6]
        box3d_pts_3d = my_compute_box_3d(centroid,
                                         np.array([l, w, h]), heading_angle)
        pc_in_box3d, inds = extract_pc_in_box3d( \
            pc, box3d_pts_3d)
        # Assign first dimension to indicate it is in an object box
        point_votes[inds, 0] = 1
        # Add the votes (all 0 if the point is not in any object's OBB)
        votes = np.expand_dims(centroid, 0) - pc_in_box3d[:, 0:3]
        sparse_inds = indices[inds]  # turn dense True,False inds to sparse number-wise inds
        for i in range(len(sparse_inds)):
            j = sparse_inds[i]
            point_votes[j, int(point_vote_idx[j] * 3 + 1):int((point_vote_idx[j] + 1) * 3 + 1)] = votes[i, :]
            # Populate votes with the fisrt vote
            if point_vote_idx[j] == 0:
                point_votes[j, 4:7] = votes[i, :]
                point_votes[j, 7:10] = votes[i, :]
        point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds] + 1)
    return point_votes
