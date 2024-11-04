import torch

def xywh2x1y1x2y2(bboxes: torch.Tensor) -> torch.Tensor:
    # convert xywh -> xyxy
    # this implementation could be done inplace, and made easier, but I need this to
    # always return a new tensor at a new memory address
    x1y1 = bboxes[..., :2] - (bboxes[..., 2:] / 2)
    x2y2 = x1y1 + bboxes[..., 2:]
    bboxes = torch.cat([x1y1, x2y2], dim=-1)
    return bboxes

def x1y1x2y22xywh(bboxes: torch.Tensor) -> torch.Tensor:
    # convert xyxy -> xywh
    # this implementation could be done inplace, and made easier, but I need this to
    # always return a new tensor at a new memory address
    wh = bboxes[..., 2:] - bboxes[..., :2]
    xy = bboxes[..., :2] + (wh / 2)
    bboxes = torch.cat([xy, wh], dim=-1)
    return bboxes

def compute_iou(preds_xywh: torch.Tensor, targets_xywh: torch.Tensor, e: float=1e-7, use_ciou: bool=False) -> torch.Tensor:
        assert (preds_xywh.ndim == targets_xywh.ndim + 1) or (preds_xywh.ndim == targets_xywh.ndim)
        assert preds_xywh.shape[-1] == targets_xywh.shape[-1] == 4
        if targets_xywh.ndim != preds_xywh.ndim:
                targets_xywh = targets_xywh.unsqueeze(dim=-2)

        preds_w = preds_xywh[..., 2:3]
        preds_h = preds_xywh[..., 3:]
        preds_x1 = preds_xywh[..., 0:1] - (preds_w / 2)
        preds_y1 = preds_xywh[..., 1:2] - (preds_h / 2)
        preds_x2 = preds_x1 + preds_w
        preds_y2 = preds_y1 + preds_h

        targets_w = targets_xywh[..., 2:3]
        targets_h = targets_xywh[..., 3:]
        targets_x1 = targets_xywh[..., 0:1] - (targets_w / 2)
        targets_y1 = targets_xywh[..., 1:2] - (targets_h / 2)
        targets_x2 = targets_x1 + targets_w
        targets_y2 = targets_y1 + targets_h

        intersection_w = (torch.min(preds_x2, targets_x2) - torch.max(preds_x1, targets_x1)).clip(min=0)
        intersection_h = (torch.min(preds_y2, targets_y2) - torch.max(preds_y1, targets_y1)).clip(min=0)
        intersection = intersection_w * intersection_h
        union = (preds_w * preds_h) + (targets_w * targets_h) - intersection
        iou = intersection / (union + e)
        if use_ciou:
            cw = (torch.max(preds_x2, targets_x2) - torch.min(preds_x1, targets_x1))
            ch = (torch.max(preds_y2, targets_y2) - torch.min(preds_y1, targets_y1))
            c2 = cw.pow(2) + ch.pow(2) + e
            v = (4 / (torch.pi**2)) * (torch.arctan(targets_w / targets_h) - torch.arctan(preds_w / preds_h)).pow(2)
            rho2 = (preds_xywh[..., :1] - targets_xywh[..., :1]).pow(2) + (preds_xywh[..., 1:2] - targets_xywh[..., 1:2]).pow(2)
            with torch.no_grad():
                a = v / (v - iou + (1 + e))
            iou = iou - ((rho2/c2) + (a * v))
        return iou.squeeze(-1)