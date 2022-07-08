import numpy as np
import sys
sys.path.insert(0,'./mytools')
from dota_kit import polyiou
from mmdet.ops import box_iou_rotated
from ..transforms_rbox import poly2rbox_torch
from .builder import IOU_CALCULATORS


@IOU_CALCULATORS.register_module()
class RBboxOverlaps2D(object):
    """2D IoU Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='riou'):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If is_aligned is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union) or iof (intersection
                over foreground).

        Returns:
            ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
        """
        assert bboxes1.size(-1) in [0, 8, 9]
        assert bboxes2.size(-1) in [0, 8, 9]
        if bboxes2.size(-1) == 9:
            bboxes2 = bboxes2[..., :8]
        if bboxes1.size(-1) == 9:
            bboxes1 = bboxes1[..., :8]
        return rbox_overlaps(bboxes1, bboxes2)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def box_iou_rotated_cpu(rboxes1, rboxes2):
    overlaps = np.zeros([len(rboxes1), len(rboxes2)])
    for index1, rbox1 in enumerate(rboxes1):
        for index2, rbox2 in enumerate(rboxes2):
            overlaps[index1, index2] = polyiou.iou_poly(polyiou.VectorDouble(rbox1), polyiou.VectorDouble(rbox2))
    return overlaps


def rbox_overlaps(rboxes1, rboxes2):
    # ious = box_iou_rotated_cpu(rboxes1.cpu().numpy().astype(np.float64), rboxes2.cpu().numpy().astype(np.float64))
    # return torch.from_numpy(ious).to(rboxes1.device)
    if rboxes1.size(1) == 8:
        rboxes1 = poly2rbox_torch(rboxes1.float())
    if rboxes2.size(1) == 8:
        rboxes2 = poly2rbox_torch(rboxes2.float())
    try:
        ious = box_iou_rotated(rboxes1, rboxes2)
    except:
        print(rboxes1)
        print(rboxes2)
        exit(0)
    return ious

