# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- focal loss (binary multi-label) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='sum'):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, preds, targets, weights=None):
        # preds: (N, C) logits, targets: (N, C) 0/1
        prob = preds.sigmoid()
        pt = torch.where(targets==1, prob, 1 - prob)
        alpha_factor = torch.where(targets==1, self.alpha, 1 - self.alpha)
        weight = alpha_factor * (1 - pt) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(preds, targets.float(), reduction='none')
        loss = weight * bce
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)
        if self.reduction == 'sum': return loss.sum()
        if self.reduction == 'mean': return loss.mean()
        return loss

# --- GIoU ---
def giou_loss(pred_boxes, target_boxes, eps=1e-7):
    # pred_boxes & target_boxes are (N,4) in xyxy pixel coords
    px1,py1,px2,py2 = pred_boxes.unbind(-1)
    gx1,gy1,gx2,gy2 = target_boxes.unbind(-1)
    ix1 = torch.max(px1, gx1); iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2); iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    area_p = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    area_g = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
    union = area_p + area_g - inter + eps
    iou = inter / union
    cx1 = torch.min(px1, gx1); cy1 = torch.min(py1, gy1)
    cx2 = torch.max(px2, gx2); cy2 = torch.max(py2, gy2)
    carea = (cx2 - cx1).clamp(0) * (cy2 - cy1).clamp(0) + eps
    giou = iou - (carea - union) / carea
    return (1 - giou).sum()

# --- IoU helper on xyxy --- 
def bbox_iou(box1, box2, eps=1e-7):
    # box1: (N,4), box2: (M,4) -> (N,M)
    N = box1.shape[0]; M = box2.shape[0]
    if N==0 or M==0:
        return torch.zeros((N,M), dtype=box1.dtype, device=box1.device)
    a_area = (box1[:,2]-box1[:,0]).clamp(0) * (box1[:,3]-box1[:,1]).clamp(0)
    b_area = (box2[:,2]-box2[:,0]).clamp(0) * (box2[:,3]-box2[:,1]).clamp(0)
    lt = torch.max(box1[:,None,:2], box2[None,:,:2])  # (N,M,2)
    rb = torch.min(box1[:,None,2:], box2[None,:,2:])  # (N,M,2)
    wh = (rb - lt).clamp(min=0)                       # (N,M,2)
    inter = wh[:,:,0] * wh[:,:,1]
    union = a_area[:,None] + b_area[None,:] - inter + eps
    return inter / union

# --- matching and loss wrapper ---
def match_anchors_to_gt(anchors_xyxy, gt_boxes_xyxy, gt_labels, pos_iou_thresh=0.5, neg_iou_thresh=0.4):
    # anchors_xyxy: (A,4) in pixels; gt_boxes_xyxy: (G,4) pixels
    A = anchors_xyxy.shape[0]
    G = gt_boxes_xyxy.shape[0]
    if G == 0:
        labels = torch.zeros((A,), dtype=torch.long, device=anchors_xyxy.device)  # background
        assigned_gt = torch.full((A,), -1, dtype=torch.long, device=anchors_xyxy.device)
        return labels, assigned_gt
    ious = bbox_iou(anchors_xyxy, gt_boxes_xyxy)  # (A,G)
    best_iou_per_anchor, best_gt_idx = ious.max(dim=1)
    labels = torch.full((A,), -1, dtype=torch.long, device=anchors_xyxy.device)  # -1 ignore
    assigned_gt = torch.full((A,), -1, dtype=torch.long, device=anchors_xyxy.device)
    pos_mask = best_iou_per_anchor >= pos_iou_thresh
    neg_mask = best_iou_per_anchor < neg_iou_thresh
    labels[neg_mask] = 0  # background (0)
    labels[pos_mask] = 1  # positive marker (1)
    assigned_gt[pos_mask] = best_gt_idx[pos_mask]
    # also ensure each GT has at least its best anchor positive (force match)
    best_anchor_for_gt = ious.argmax(dim=0)  # (G,)
    labels[best_anchor_for_gt] = 1
    assigned_gt[best_anchor_for_gt] = torch.arange(G, device=anchors_xyxy.device)
    return labels, assigned_gt

def compute_losses(model_outputs, anchors_norm, targets, input_size, num_classes,
                   focal_alpha=0.25, focal_gamma=2.0, lambda_box=2.0):
    """
    model_outputs: list of (cls_logits, reg, obj) per level (shapes: (B, A_lvl*num_classes, H, W), etc.)
    anchors_norm: (A_total, 4) normalized cxcywh (0..1)
    targets: list length B of {'boxes': (G,4) pixels, 'labels': (G,) ints}
    Returns: scalar loss (tensor), dict of loss components
    """
    device = anchors_norm.device
    B = model_outputs[0][0].shape[0]
    # flatten outputs to (B, A_total, ...)
    cls_list=[]; reg_list=[]; obj_list=[]
    for cls, reg, obj in model_outputs:
        # cls shape (B, A_lvl*num_classes, H, W) -> convert to (B, A_lvl, num_classes)
        b, c, H, W = cls.shape
        cls = cls.permute(0,2,3,1).contiguous().view(b, -1, cls.shape[1]//( (H*W) if False else cls.shape[1]))  # fallback
        cls = cls.permute(0,2,1) if False else cls  # no-op safe
        # simpler reliable reshape:
        cls = cls.permute(0,2,3,1).contiguous().view(b, -1, cls.shape[-1])
        # regression: (B, A_lvl*4, H, W) -> (B, A_lvl, 4)
        reg = reg.permute(0,2,3,1).contiguous().view(b, -1, 4)
        obj = obj.permute(0,2,3,1).contiguous().view(b, -1, 1)
        cls_list.append(cls); reg_list.append(reg); obj_list.append(obj)
    cls_all = torch.cat(cls_list, dim=1)   # (B, A, num_classes)
    reg_all = torch.cat(reg_list, dim=1)   # (B, A, 4)  (we assume network outputs normalized cxcywh directly)
    obj_all = torch.cat(obj_list, dim=1)   # (B, A, 1)
    A = cls_all.shape[1]

    # anchors in pixels
    anchors_xyxy = cxcywh_to_xyxy(anchors_norm.to(device), input_size)  # (A,4)

    focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='sum')

    total_cls_loss = 0.0; total_box_loss = 0.0
    total_pos = 0
    for i in range(B):
        gt = targets[i]
        gt_boxes = gt['boxes'].to(device)   # (G,4) pixels
        gt_labels = gt['labels'].to(device) # (G,)
        labels_assign, assigned_gt = match_anchors_to_gt(anchors_xyxy, gt_boxes, gt_labels)
        # build classification targets (one-hot)
        cls_target = torch.zeros((A, num_classes), device=device)
        pos_mask = labels_assign == 1
        neg_mask = labels_assign == 0
        ignore_mask = labels_assign == -1
        if pos_mask.sum() > 0:
            matched_idxs = assigned_gt[pos_mask]  # indices into GT
            # set class one-hot for positives
            cls_target[pos_mask, gt_labels[matched_idxs]] = 1.0
        # compute classification loss only on non-ignored anchors
        cls_pred = cls_all[i]  # (A, num_classes) logits
        cls_weights = (~ignore_mask).float()  # 1 for used anchors
        cls_loss = focal(cls_pred, cls_target, weights=cls_weights)
        total_cls_loss += cls_loss

        # box loss for positives
        if pos_mask.sum() > 0:
            total_pos += int(pos_mask.sum())
            # predicted reg for positive anchors (we assume network predicts normalized cxcywh)
            pred_regs = reg_all[i][pos_mask]        # (P,4) normalized
            pred_boxes = cxcywh_to_xyxy(pred_regs, input_size)  # (P,4) pixels
            gt_for_pos = gt_boxes[ assigned_gt[pos_mask] ]      # (P,4) pixels
            box_loss = giou_loss(pred_boxes, gt_for_pos)
            total_box_loss += box_loss
        else:
            total_box_loss += 0.0

    # normalize
    if total_pos == 0:
        loss = total_cls_loss + lambda_box * total_box_loss
        return loss, {"cls": float(total_cls_loss), "box": float(total_box_loss), "pos": 0}
    loss = (total_cls_loss / B) + lambda_box * (total_box_loss / B)
    return loss, {"cls": float(total_cls_loss / B), "box": float(total_box_loss / B), "pos": total_pos}

# --- util: convert normalized cxcywh -> xyxy pixels ---
def cxcywh_to_xyxy(boxes, input_size):
    b = boxes.clone()
    b[...,0] = (boxes[...,0] - boxes[...,2]/2) * input_size
    b[...,1] = (boxes[...,1] - boxes[...,3]/2) * input_size
    b[...,2] = (boxes[...,0] + boxes[...,2]/2) * input_size
    b[...,3] = (boxes[...,1] + boxes[...,3]/2) * input_size
    return b
