#!/usr/bin/env bash
cd pointnet2
python setup.py install
cd ../utils/iou3d
python setup.py install
cd ../roipool3d
python setup.py install
cd ..
