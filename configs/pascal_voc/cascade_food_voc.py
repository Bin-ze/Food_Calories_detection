_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn_food.py',
    '../_base_/datasets/Food_voc.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
load_from='/home/guozebin/mmdetection/tools/work_dirs/cascade_food_voc/epoch_1.pth'