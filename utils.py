# utils.py
import torch, os, math
from torchvision.ops import nms, box_iou

def move_batch_to_device(batch, device):
    images, targets = batch
    images = images.to(device, non_blocking=True)
    new_targets=[]
    for t in targets:
        t_on = {}
        t_on["boxes"] = t["boxes"].to(device) if t["boxes"].numel()>0 else t["boxes"].to(device)
        t_on["labels"] = t["labels"].to(device) if t["labels"].numel()>0 else t["labels"].to(device)
        new_targets.append(t_on)
    return images, new_targets

def build_anchors_for_model(model, input_size=640, ratios=[0.5,1.0,2.0]):
    device = next(model.parameters()).device
    sizes = model.feature_map_sizes((input_size,input_size))
    anchors=[]
    base_scales = getattr(model, "base_scales", [0.04,0.12,0.36])
    for (H,W), base_scale in zip(sizes, base_scales):
        for i in range(H):
            for j in range(W):
                cy = (i + 0.5)/H; cx = (j+0.5)/W
                for r in ratios[:model.num_anchors]:
                    area = base_scale*base_scale
                    w = math.sqrt(area*r); h = math.sqrt(area/r)
                    anchors.append([cx,cy,w,h])
    return torch.tensor(anchors, dtype=torch.float32, device=device)

def cxcywh_to_xyxy(boxes, input_size):
    b = boxes.clone()
    b[...,0] = (boxes[...,0] - boxes[...,2]/2) * input_size
    b[...,1] = (boxes[...,1] - boxes[...,3]/2) * input_size
    b[...,2] = (boxes[...,0] + boxes[...,2]/2) * input_size
    b[...,3] = (boxes[...,1] + boxes[...,3]/2) * input_size
    return b

def decode_outputs(output_list, anchors_norm, input_size=640, conf_thresh=0.05, iou_thres=0.5):
    """
    Robust decoder.
    - output_list: list of (cls_logits, reg, obj) per FPN level
      cls_logits: (B, A_lvl * num_classes, H, W)
      reg: (B, A_lvl * 4, H, W)    (we assume normalized cxcywh)
      obj: (B, A_lvl * 1, H, W)
    - anchors_norm: (A_total,4) normalized cxcywh (not used to decode reg but used for size)
    """
    B = output_list[0][0].shape[0]
    cls_list = []; reg_list = []; obj_list = []

    for cls, reg, obj in output_list:
        # cls/reg/obj are 4D: (B, C, H, W)
        if cls.dim() != 4 or reg.dim() != 4 or obj.dim() != 4:
            raise RuntimeError("Expected 4D tensors in decoder")

        b, c_cls, H, W = cls.shape
        _, c_reg, _, _ = reg.shape
        _, c_obj, _, _ = obj.shape

        # convert to (B, H, W, C)
        cls = cls.permute(0, 2, 3, 1).contiguous()   # (B, H, W, c_cls)
        reg = reg.permute(0, 2, 3, 1).contiguous()   # (B, H, W, c_reg)
        obj = obj.permute(0, 2, 3, 1).contiguous()   # (B, H, W, c_obj)

        # anchors per cell for this level
        if (c_reg % 4) != 0:
            raise RuntimeError(f"Unexpected reg channels {c_reg}; must be divisible by 4")
        A_lvl = c_reg // 4

        # num_classes per anchor
        if (c_cls % A_lvl) != 0:
            raise RuntimeError(f"Cannot infer num_classes: c_cls={c_cls}, A_lvl={A_lvl}")
        num_classes = c_cls // A_lvl

        # reshape to (B, A_lvl * H * W, num_classes) etc.
        cls = cls.view(b, H * W * A_lvl, num_classes)
        reg = reg.view(b, H * W * A_lvl, 4)
        obj = obj.view(b, H * W * A_lvl, 1)

        cls_list.append(cls); reg_list.append(reg); obj_list.append(obj)

    # concat across pyramid levels
    cls_all = torch.cat(cls_list, dim=1)   # (B, A_total, num_classes)
    reg_all = torch.cat(reg_list, dim=1)   # (B, A_total, 4)
    obj_all = torch.cat(obj_list, dim=1)   # (B, A_total, 1)

    anchors_xyxy = cxcywh_to_xyxy(anchors_norm, input_size)
    results = []
    for b in range(B):
        scores = (cls_all[b].sigmoid() * obj_all[b].sigmoid())  # (A, num_classes)
        max_scores, labels = scores.max(dim=1)
        keep = max_scores > conf_thresh
        if keep.sum() == 0:
            results.append([])
            continue
        idx = keep.nonzero(as_tuple=False).squeeze(1)
        chosen_scores = max_scores[idx]; chosen_labels = labels[idx]
        chosen_regs = reg_all[b][idx]  # normalized cxcywh
        chosen_boxes = cxcywh_to_xyxy(chosen_regs, input_size)
        detections = []
        for cls_id in torch.unique(chosen_labels):
            mask = chosen_labels == cls_id
            if mask.sum() == 0: continue
            boxes_cls = chosen_boxes[mask]; scores_cls = chosen_scores[mask]
            keep_idx = nms(boxes_cls, scores_cls, iou_thres)
            for k in keep_idx:
                detections.append((boxes_cls[k].cpu(), scores_cls[k].item(), int(cls_id.item())))
        results.append(detections)
    return results


# simple mAP50 evaluator (same as earlier but concise)
def compute_map50(predictions, targets, num_classes=24, iou_threshold=0.5):
    import numpy as np
    aps=[]; per_class_ap={}
    for cls in range(num_classes):
        tp_scores=[]; total_gts=0
        for preds, tgt in zip(predictions, targets):
            preds_cls = [p for p in preds if p[2]==cls]
            gt_mask = (tgt['labels'].cpu().numpy()==cls)
            gt_boxes = tgt['boxes'].cpu() if gt_mask.sum()>0 else torch.zeros((0,4))
            total_gts += int(gt_mask.sum())
            preds_cls = sorted(preds_cls, key=lambda x: x[1], reverse=True)
            matched = torch.zeros(len(gt_boxes), dtype=torch.bool) if gt_boxes.numel()>0 else torch.zeros((0,), dtype=torch.bool)
            for box_pred, score, _ in preds_cls:
                if gt_boxes.numel()==0:
                    tp_scores.append((0, score)); continue
                ious = box_iou(box_pred.unsqueeze(0).cpu(), gt_boxes.cpu())
                iou_vals, inds = ious.max(dim=1)
                iou_val = iou_vals.item(); ind = inds.item()
                if iou_val >= iou_threshold and not matched[ind]:
                    matched[ind] = True; tp_scores.append((1, score))
                else:
                    tp_scores.append((0, score))
        if total_gts == 0: per_class_ap[cls] = None; continue
        if len(tp_scores) == 0: per_class_ap[cls]=0.0; aps.append(0.0); continue
        tp_scores = sorted(tp_scores, key=lambda x: x[1], reverse=True)
        tps = [x[0] for x in tp_scores]; fps = [1-x for x in tps]
        tps_cum = np.cumsum(tps); fps_cum = np.cumsum(fps)
        recalls = tps_cum / (total_gts + 1e-8)
        precisions = tps_cum / (tps_cum + fps_cum + 1e-8)
        for i in range(len(precisions)-2, -1, -1):
            if precisions[i] < precisions[i+1]: precisions[i]=precisions[i+1]
        recall_vals = [0.0] + recalls.tolist() + [1.0]
        precision_vals = [precisions[0]] + precisions.tolist() + [0.0]
        ap=0.0
        for i in range(len(recall_vals)-1):
            ap += (recall_vals[i+1] - recall_vals[i]) * precision_vals[i]
        per_class_ap[cls]=ap; aps.append(ap)
    valid = [a for a in aps if a is not None]
    mAP50 = float(sum(valid)/len(valid)) if len(valid)>0 else 0.0
    return mAP50, per_class_ap

def save_checkpoint(state, ckpt_dir, epoch, is_best=False):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch}.pth")
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(ckpt_dir, "best_model.pth"))
