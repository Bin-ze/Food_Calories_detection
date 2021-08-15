import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmdet.models.builder import HEADS


@HEADS.register_module()
class Segmentation_module(BaseModule):

    def __init__(self,
                 ignore_label=255,
                 loss_weight=0.2,
                 init_cfg=None):
        super(Segmentation_module, self).__init__(init_cfg)
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.fp16_enabled = False
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)


    @auto_fp16()
    def forward(self, feats):
        return feats
    def forward_train(self,
                      x,
                      gt_semantic_seg):
        losses=dict()
        mask_pred=self.forward(x)
        loss_seg=self.loss(mask_pred,gt_semantic_seg)
        losses['loss_semantic_seg']=loss_seg
        return losses, mask_pred

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, labels):
        labels = labels.squeeze(1).long()
        loss_semantic_seg = self.criterion(mask_pred, labels)
        loss_semantic_seg *= self.loss_weight
        return loss_semantic_seg
