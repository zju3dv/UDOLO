import sys
sys.path.append('/home/xieyiming/repo/votenetplus/votenet')
from models.ap_helper import APCalculator
import _pickle as cPickle
from scannet.model_util_scannet import ScannetSVDatasetConfig
import numpy as np
import os
from box_util import get_3d_box

DC = ScannetSVDatasetConfig()


def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[..., 1] *= -1
    return pc2


DATASET_CONFIG = ScannetSVDatasetConfig()

gt_data_path = '/data/scannet/votenet_files/scannet_train_detection_data'

file = 'eval_scannet_single_aabb_early'
gt_path = '{}/gt.pkl'.format(file)
data_path = '{}/oriented_backend_resultpredict.pkl'.format(file)  # for no opt back frontend
# data_path = '../eval_scannet/no_feed_frontend_corners_predict.pkl'  # for no opt back frontend
# data_path = '../eval_scannet/no_feed_backend_corners_predict.pkl'  # for no opt back frontend
# data_path = '../eval_scannet/no_opt_frontend_corners_predict.pkl'  # for no opt back frontend
print(file)

with open(gt_path, 'rb') as f:
    gts = cPickle.load(f)

with open(data_path, 'rb') as f:
    predictions = cPickle.load(f)

roidb_filename = '/home/xieyiming/repo/votenetplus/votenet/scannet_single_aabb/keyframes_val_valid.pkl'
with open(roidb_filename, 'rb') as f:
    gts_loader = cPickle.load(f)
    gts_loader = gts_loader

predictions_ = {}
scene_id = 0
gts_ = {}
scene_name = None
scene_predictions = {}
for i, gts_l in enumerate(gts_loader):
    sn = gts_l['scene']
    if sn != scene_name:
        if scene_name is None:
            scene_name = sn
        if len(scene_predictions) != 0:
            predictions_[scene_id] = list(scene_predictions.values())
            gts_[scene_id] = batch_gt_map_cls
            batch_gt_map_cls = []
            scene_predictions = {}
            scene_name = sn
            scene_id += 1

        gt_box = np.load(os.path.join(gt_data_path, scene_name) + '_bbox.npy')
        K2 = gt_box.shape[0]  # K2==MAX_NUM_OBJ
        gt_corners_3d_upright_camera = np.zeros((K2, 8, 3))
        gt_box[:, 0:3] = flip_axis_to_camera(gt_box[:, 0:3])
        label = [np.where(DC.nyu40ids == x)[0][0] for x in gt_box[:, -1]]
        for j in range(K2):
            corners_3d_upright_camera = get_3d_box(gt_box[j, 3:6], np.zeros_like(gt_box[j, :1]), gt_box[j, :3])
            gt_corners_3d_upright_camera[j] = corners_3d_upright_camera
        batch_gt_map_cls = [(label[j], gt_corners_3d_upright_camera[j], 0.0) for j in
                            range(gt_corners_3d_upright_camera.shape[0])]

    for pred in predictions[i]:
        scene_predictions[str(pred[3]) + '_' + str(pred[0])] = pred

# print(predictions_[0])
print(len(predictions_))
# print(predictions[0])
predictions = predictions_
gts = gts_

ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
                      for iou_thresh in [0.25, 0.5]]

for ap_calculator in ap_calculator_list:
    ap_calculator.load(gts, predictions)

# Evaluate average precision
for i, ap_calculator in enumerate(ap_calculator_list):
    print('-' * 10, 'iou_thresh: %f' % ([0.25, 0.5][i]), '-' * 10)
    metrics_dicts = ap_calculator.compute_metrics()
    difficulty = ['easy', 'moderate', 'hard']
    # difficulty = ['occluded']
    for metrics_dict, diff in zip(metrics_dicts, difficulty):
        for key in metrics_dict:
            print('%s eval backend %s: %f' % (diff, key, metrics_dict[key]))
