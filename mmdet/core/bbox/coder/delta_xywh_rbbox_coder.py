import numpy as np
import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class DeltaXYWHRBBoxCoder(BaseBBoxCoder):
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
                 target_means=(0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1.)):
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
        assert gt_bboxes.size(-1) == 8
        encoded_bboxes = rbbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               err=0.1):
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
        decoded_bboxes = delta2rbbox(bboxes, pred_bboxes, self.means, self.stds,
                                    max_shape, wh_ratio_clip, err=err)

        return decoded_bboxes

    def decode_nms(self,
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
        decoded_bboxes = delta2bbox_nms(bboxes, pred_bboxes, self.means, self.stds,
                                    max_shape)

        return decoded_bboxes

    def decode_comparetarget(self,
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
        decoded_bboxes = delta2comp_target(bboxes, pred_bboxes, self.means, self.stds,
                                    max_shape)

        return decoded_bboxes

    def decode_bbox(self,
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

def rbbox2delta(proposals, gt, means=(0., 0., 0., 0.), stds=(1., 1., 1., 1.)):
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
    assert proposals.size(0) == gt.size(0)

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    x_all, _ = torch.sort(gt[..., 0:8:2])
    y_all, _ = torch.sort(gt[..., 1:8:2])

    gx1 = (x_all[..., 0] + x_all[..., 2]) * 0.5
    gy1 = (y_all[..., 0] + y_all[..., 2]) * 0.5
    gw1 = x_all[..., 2] - x_all[..., 0]
    gh1 = y_all[..., 2] - y_all[..., 0]
    gx2 = (x_all[..., 1] + x_all[..., 3]) * 0.5
    gy2 = (y_all[..., 1] + y_all[..., 3]) * 0.5
    gw2 = x_all[..., 3] - x_all[..., 1]
    gh2 = y_all[..., 3] - y_all[..., 1]

    dx1 = (gx1 - px) / pw
    dy1 = (gy1 - py) / ph
    dw1 = torch.log(gw1.clamp(min=1e-6) / pw)
    dh1 = torch.log(gh1.clamp(min=1e-6) / ph)
    dx2 = (gx2 - px) / pw
    dy2 = (gy2 - py) / ph
    dw2 = torch.log(gw2.clamp(min=1e-6) / pw)
    dh2 = torch.log(gh2.clamp(min=1e-6) / ph)
    deltas = torch.stack([dx1, dy1, dw1, dh1, dx2, dy2, dw2, dh2], dim=-1)

    means = deltas.new_tensor(means).repeat(8 // 4).unsqueeze(0)
    stds = deltas.new_tensor(stds).repeat(8 // 4).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas


def delta2rbbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               err=0.1):
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
    means = deltas.new_tensor(means).view(1, -1).repeat(1, 8 // 4)
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, 8 // 4)
    denorm_deltas = deltas[..., :8] * stds + means
    obliquity_factor = deltas[..., 8].clone()
    direction_factor_top = deltas[..., -2].clone()
    direction_factor_left = deltas[..., -1].clone()

    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dh)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    gy = py + ph * dy
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5

    x_raw = torch.stack([x1[:,0], x1[:,1], x2[:,0], x2[:,1]], dim=-1)
    y_raw = torch.stack([y1[:,0], y1[:,1], y2[:,0], y2[:,1]], dim=-1)
    x = x_raw.new_zeros(x_raw.shape)
    y = y_raw.new_zeros(y_raw.shape)

    # change_index = (((x_raw[:,1] - x_raw[:,0])**2 - 0.5 * (x_raw[:,2] - x_raw[:,1])**2 - (y_raw[:,1] - y_raw[:,0])**2 > 0) & (obliquity_factor > 0.5)).nonzero()
    # direction_factor[change_index] = 1 - direction_factor[change_index]

    # index = (direction_factor > 0.5).nonzero()
    # x[index.squeeze(), :] = x_raw[index, [1,0,2,3]]
    # index = (direction_factor <= 0.5).nonzero()
    # x[index.squeeze(), :] = x_raw[index, [2,0,1,3]]
    # index = (((obliquity_factor > 0.5) & (direction_factor > 0.5)) | ((obliquity_factor <= 0.5) & (direction_factor <= 0.5))).nonzero()
    # y[index.squeeze(), :] = y_raw[index, [0,2,3,1]]
    # index = (((obliquity_factor > 0.5) & (direction_factor <= 0.5)) | ((obliquity_factor <= 0.5) & (direction_factor > 0.5))).nonzero()
    # y[index.squeeze(), :] = y_raw[index, [0,1,3,2]]

    change_index = ((obliquity_factor > 0.5) & \
            (((direction_factor_left - 0.5) * (direction_factor_top - 0.5)) > 0)).nonzero()
    left = (torch.abs(direction_factor_top[change_index] - 0.5) > torch.abs(direction_factor_left[change_index] - 0.5))
    top = ~left
    direction_factor_left[change_index[left]] = 1 - direction_factor_left[change_index[left]]
    direction_factor_top[change_index[top]] = 1 - direction_factor_top[change_index[top]]

    # change_index = ((obliquity_factor < 0.5) & \
    #         (((direction_factor_left - 0.5) * (direction_factor_top - 0.5)) < 0)).nonzero()
    # left = (torch.abs(direction_factor_top[change_index] - 0.5) > torch.abs(direction_factor_left[change_index] - 0.5))
    # top = ~left
    # direction_factor_left[change_index[left]] = 1 - direction_factor_left[change_index[left]]
    # direction_factor_top[change_index[top]] = 1 - direction_factor_top[change_index[top]]

    index = (direction_factor_left > 0.5).nonzero()
    y[index.squeeze(), :] = y_raw[index, [0,1,3,2]]
    index = (direction_factor_left <= 0.5).nonzero()
    y[index.squeeze(), :] = y_raw[index, [0,2,3,1]]
    index = (direction_factor_top > 0.5).nonzero()
    x[index.squeeze(), :] = x_raw[index, [1,0,2,3]]
    index = (direction_factor_top <= 0.5).nonzero()
    x[index.squeeze(), :] = x_raw[index, [2,0,1,3]]

    bboxes = torch.stack([x[..., 0], y[..., 0], x[..., 1], y[..., 1], x[..., 2], y[..., 2], x[..., 3], y[..., 3]],
                         dim=-1).view(denorm_deltas.size())
    return bboxes

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
    means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas[..., :8] * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]

    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dx)
    ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dy)
    # Use exp(network energy) to enlarge/shrink each roi

    gx = px + pw * dx
    gy = py + ph * dy
    gw = pw * dw.exp()
    gh = ph * dh.exp()

    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5

    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        y2 = y2.clamp(min=0, max=max_shape[0])
    # bboxes = torch.stack([x1[..., 0], y1[..., 0], x2[..., 0], y2[..., 0], x1[..., 1], y1[..., 1], x2[..., 1], y2[..., 1]],
    #                      dim=-1).view(-1,4)
    bboxes = torch.stack([x1[..., 0], y1[..., 0], x2[..., 0], y2[..., 0], x1[..., 1], y1[..., 1], x2[..., 1], y2[..., 1],
                         x1[..., 0], y1[..., 0], x2[..., 1], y2[..., 1]],
                         dim=-1).view(-1,4)
    return bboxes


def delta2bbox_nms(rois,
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
    means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas[..., :8] * stds + means
    obliquity_factor = deltas[..., 8].clone()
    obliquity_factor = torch.sqrt(obliquity_factor)
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]

    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dx)
    ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dy)
    # Use exp(network energy) to enlarge/shrink each roi

    gx = px + pw * dx
    gy = py + ph * dy
    gw = pw * dw.exp()
    gh = ph * dh.exp()

    # w = (gw[..., 0] + gw[..., 1]) * 0.5
    # h = (gh[..., 0] + gh[..., 1]) * 0.5
    # w_max = gx[..., 1] - gx[..., 0] + w
    # h_max = gy[..., 1] - gy[..., 0] + h

    # alpha = w / w_max
    # beta = h / h_max
    # index = direction_factor[:,0] > 0.5
    # alpha[index] = 1 - alpha[index]
    # index = direction_factor[:,1] > 0.5
    # beta[index] = 1 - beta[index]

    # obliquity_factor = alpha + beta - 2 * alpha * beta
    # # print(obliquity_factor)

    cx = (gx[..., 0] + gx[..., 1]) * 0.5
    cy = (gy[..., 0] + gy[..., 1]) * 0.5
    w = (gw[..., 0] + gw[..., 1]) * 0.5 * obliquity_factor
    h = (gh[..., 0] + gh[..., 1]) * 0.5 * obliquity_factor

    x1 = (cx - w * 0.5).unsqueeze(1)
    y1 = (cy - h * 0.5).unsqueeze(1)
    x2 = (cx + w * 0.5).unsqueeze(1)
    y2 = (cy + h * 0.5).unsqueeze(1)

    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        y2 = y2.clamp(min=0, max=max_shape[0])
    bboxes = torch.stack([x1, y1, x2, y2],
                         dim=-1).view(-1,4)
    return bboxes

def delta2comp_target(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None):

    means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas[..., :8] * stds + means
    obliquity_factor = deltas[..., -3]
    direction_factor1 = deltas[..., -2]
    direction_factor2 = deltas[..., -1]
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]

    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dx)
    ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dy)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    gy = py + ph * dy
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        y2 = y2.clamp(min=0, max=max_shape[0])
    
    x_raw = torch.stack([x1[:,0], x1[:,1], x2[:,0], x2[:,1]], dim=-1)
    y_raw = torch.stack([y1[:,0], y1[:,1], y2[:,0], y2[:,1]], dim=-1)
    x21 = (x_raw[:,2] - x_raw[:,1]) / (x_raw[:,3] - x_raw[:,0])
    y21 = (y_raw[:,2] - y_raw[:,1]) / (y_raw[:,3] - y_raw[:,0])
    result = torch.stack([obliquity_factor, direction_factor1, direction_factor2, torch.abs(x21 * y21)], dim=-1)
    return result