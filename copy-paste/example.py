import cv2
import numpy as np
from copy_paste import CopyPaste
from coco import CocoDetectionCP
from visualize import display_instances
import albumentations as A
import random
from matplotlib import pyplot as plt
#%%

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y =  np.zeros_like(x)
    y[:, 0] = x[:, 0]  # top left x
    y[:, 1] = x[:, 1]  # top left y
    y[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    y[:, 4] = x[:,4]
    return y
transform = A.Compose([
    A.RandomScale(scale_limit=(-0.9, 1), p=1), #LargeScaleJitter from scale of 0.1 to 2
    A.PadIfNeeded(512, 512, border_mode=0), #pads with image in the center, not the top left like the paper
    A.RandomCrop(512, 512),
    CopyPaste(blend=True, sigma=1, pct_objects_paste=0.8, p=1.) #pct_objects_paste is a guess
], bbox_params=A.BboxParams(format="coco", min_visibility=0.05)
)

data = CocoDetectionCP(
    '../data/foof_full_coco/',
    '../data/foof_full_coco/annotations.json',
    transform
)

#%%
path='../copy-paste-aug/fake_image/'
for i in range(100):
    #f, ax = plt.subplots(1, 1, figsize=(16, 16))
    index = random.randint(0, len(data)-1)
    img_data = data[index]
    image = img_data['image']
    masks = img_data['masks']
    bboxes = img_data['bboxes']
    bbox=xywh2xyxy(np.array(bboxes)[:,:5]).astype(int)

    cv2.imwrite(path+'{}.jpg'.format(i),cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    with open('lable.txt','a',encoding='utf-8') as f:
        f.write(path+'{}.jpg'.format(i)+'')
        for i in bbox:
            f.write(" " + ",".join([str(a) for a in i[:5]]))
        f.write('\n')

    # empty = np.array([])
    # #display_instances(image, empty, empty, empty, empty, show_mask=False, show_bbox=False, ax=ax)
    # f1, ax1 = plt.subplots(1, 1, figsize=(16, 16))
    # if len(bboxes) > 0:
    #     boxes = np.stack([b[:4] for b in bboxes], axis=0)
    #     box_classes = np.array([b[-2] for b in bboxes])
    #     mask_indices = np.array([b[-1] for b in bboxes])
    #     show_masks = np.stack(masks, axis=-1)[..., mask_indices]
    #     class_names = {k: data.coco.cats[k]['name'] for k in data.coco.cats.keys()}
        #display_instances(image, boxes, show_masks, box_classes, class_names, show_bbox=True, ax=ax1)
    #else:
        #display_instances(image, empty, empty, empty, empty, show_mask=False, show_bbox=False, ax=ax1)
