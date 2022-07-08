import numpy as np
import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class DeltaXYXYBBoxCoder(BaseBBoxCoder):
    """Delta XYWH BBox coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh) and
    decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1., 1., 1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 4
        assert gt_bboxes.size(-1) == 8 or gt_bboxes.size(-1) == 5
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            max_shape (tuple[int], optional): Maximum shape of boxes.
                Defaults to None.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """

        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                    max_shape)

        return decoded_bboxes


def bbox2delta(proposals, gt, means=(0., 0., 0., 0.), stds=(1., 1., 1., 1.)):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 4)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """
    assert proposals.size()[0] == gt.size()[0]

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    # dx = (gt[..., 0:8:2] - px.unsqueeze(1)) / pw.unsqueeze(1)
    # dy = (gt[..., 1:8:2] - py.unsqueeze(1)) / ph.unsqueeze(1)
    #
    # deltas = torch.stack([dx[..., 0], dy[..., 0], dx[..., 1], dy[..., 1], dx[..., 2], dy[..., 2], dx[..., 3], dy[..., 3]], dim=-1)
    if gt.size(1) == 8:
        dx1 = (gt[..., 0] - px) / pw
        dy1 = (gt[..., 1] - py) / ph
        dx2 = (gt[..., 2] - px) / pw
        dy2 = (gt[..., 3] - py) / ph
        dx3 = (gt[..., 4] - px) / pw
        dy3 = (gt[..., 5] - py) / ph
        dx4 = (gt[..., 6] - px) / pw
        dy4 = (gt[..., 7] - py) / ph
        dx1x2 = (dx1 + dx2) / 2
        dy1y2 = (dy1 + dy2) / 2
        dx2x3 = (dx2 + dx3) / 2
        dy2y3 = (dy2 + dy3) / 2
        dx3x4 = (dx3 + dx4) / 2
        dy3y4 = (dy3 + dy4) / 2
        dx4x1 = (dx4 + dx1) / 2
        dy4y1 = (dy4 + dy1) / 2
        dcenterx = (dx1 + dx2 + dx3 + dx4) / 4
        dcentery = (dy1 + dy2 + dy3 + dy4) / 4
        # deltas = torch.stack((dy1, dx1,
        #                       dy4y1, dx4x1,
        #                       dy4, dx4,
        #                       dy1y2, dx1x2,
        #                       dcentery, dcenterx,
        #                       dy3y4, dx3x4,
        #                       dy2, dx2,
        #                       dy2y3, dx2x3,
        #                       dy3, dx3), -1)
        deltas = torch.stack((dx1, dy1,
                            dx2, dy2,
                            dx3, dy3,
                            dx4, dy4), -1)
    elif gt.size(1) == 5:
        dx = (gt[..., 0] - px) / pw
        dy = (gt[..., 1] - py) / ph
        dw = torch.log(gt[..., 2] / pw)
        dh = torch.log(gt[..., 3] / ph)
        dtheta = gt[..., 4] * np.pi / 180.0
        deltas = torch.stack((dx, dy, dw, dh, dtheta), -1)
    
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 4 * num_classes). Note N = num_anchors * W * H when
            rois is a grid of anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (tuple[int, int]): Maximum bounds for boxes. specifies (H, W)
        wh_ratio_clip (float): Maximum aspect ratio for boxes.

    Returns:
        Tensor: Boxes with shape (N, 4), where columns represent
            tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> delta2bbox(rois, deltas, max_shape=(32, 32))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.1409, 0.1409, 2.8591, 2.8591],
                [0.0000, 0.3161, 4.1945, 0.6839],
                [5.0000, 5.0000, 5.0000, 5.0000]])
    """
    if deltas.size(1) == 8:
        means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 8)
        stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 8)
        denorm_deltas = deltas * stds + means
        dx = denorm_deltas[:, 0::2]
        dy = denorm_deltas[:, 1::2]

        # Compute center of each roi
        px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
        py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
        # Compute width/height of each roi
        pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dx)
        ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dy)
        # Use exp(network energy) to enlarge/shrink each roi

        gx = px + pw * dx
        gy = py + ph * dy

        if max_shape is not None:
            gx = gx.clamp(min=0, max=max_shape[1])
            gy = gy.clamp(min=0, max=max_shape[0])
        bboxes = torch.stack([gx[..., 0], gy[..., 0], gx[..., 1], gy[..., 1], gx[..., 2], gy[..., 2], gx[..., 3], gy[..., 3]],
                            dim=-1).view(deltas.size())
    elif deltas.size(1) == 5:
        means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 5)
        stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 5)
        denorm_deltas = deltas * stds + means
        dx = denorm_deltas[:, 0]
        dy = denorm_deltas[:, 1]
        dw = denorm_deltas[:, 2]
        dh = denorm_deltas[:, 3]
        dtheta = denorm_deltas[:, 4]
        # Compute center of each roi
        px = ((rois[:, 0] + rois[:, 2]) * 0.5)
        py = ((rois[:, 1] + rois[:, 3]) * 0.5)
        # Compute width/height of each roi
        pw = (rois[:, 2] - rois[:, 0])
        ph = (rois[:, 3] - rois[:, 1])
        
        gx = px + pw * dx
        gy = py + ph * dy
        gw = pw * dw.exp()
        gh = ph * dh.exp()
        gtheta = dtheta

        bboxes = torch.stack([gx, gy, gw, gh, gtheta], dim=-1).view(deltas.size())
        bboxes = rbox2poly_torch(bboxes)
    return bboxes

def rbox2poly_torch(rboxes):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = -width*0.5, -height*0.5, width*0.5, height*0.5

    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y,
                         br_y, br_y], dim=0).reshape(2, 4, N).permute(2, 0, 1)

    sin, cos = torch.sin(angle), torch.cos(angle)
    # M.shape=[N,2,2]
    M = torch.stack([cos, -sin, sin, cos],
                    dim=0).reshape(2, 2, N).permute(2, 0, 1)
    # polys:[N,8]
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)

    return polys