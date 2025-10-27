# train.py
import os, time, argparse
from datetime import datetime
from tqdm import tqdm
import torch, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from drive_dataloader import make_loaders
from drive_object_detection_model import DriveSingleShotModel
from utils import move_batch_to_device, build_anchors_for_model, decode_outputs, compute_map50, save_checkpoint
from losses import compute_losses

def train_one_epoch(model, loader, optimizer, device, epoch, anchors, num_classes, writer=None):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train {epoch}")
    for i, (images, targets) in pbar:
        images, targets = move_batch_to_device((images, targets), device)
        outputs = model(images)
        loss, parts = compute_losses(outputs, anchors, targets, model.input_size, num_classes)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running_loss += loss.item()
        if writer and (i % 50 == 0):
            step = epoch * len(loader) + i
            writer.add_scalar("train/loss_step", loss.item(), step)
            writer.add_scalar("train/cls_loss", parts['cls'], step)
            writer.add_scalar("train/box_loss", parts['box'], step)
        pbar.set_postfix({"loss": f"{running_loss/(i+1):.4f}", "pos": parts['pos']})
    return running_loss / len(loader) if len(loader)>0 else 0.0

def evaluate(model, loader, device, anchors, num_classes, conf_thresh=0.05, iou_thres=0.5):
    model.eval()
    preds=[]; gts=[]
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader), desc="Eval")
        for i, (images, targets) in pbar:
            images, targets = move_batch_to_device((images, targets), device)
            outputs = model(images)
            batch_preds = decode_outputs(outputs, anchors, input_size=model.input_size, conf_thresh=conf_thresh, iou_thres=iou_thres)
            preds.extend(batch_preds); gts.extend(targets)
    mAP50, per_class_ap = compute_map50(preds, gts, num_classes=num_classes, iou_threshold=0.5)
    return mAP50, per_class_ap

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    train_loader, val_loader, test_loader = make_loaders(args.data_root, batch_size=args.batch_size,
                                                         input_size=args.input_size, num_workers=args.num_workers)
    model = DriveSingleShotModel(num_classes=args.num_classes, input_size=args.input_size).to(device)
    anchors = build_anchors_for_model(model, input_size=args.input_size)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_step, gamma=args.gamma)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(args.log_dir, now))
    best_map = 0.0
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, anchors, args.num_classes, writer)
        val_map, _ = evaluate(model, val_loader, device, anchors, args.num_classes)
        scheduler.step()
        t1 = time.time()
        print(f"Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | Val mAP50 {val_map:.4f} | Time {(t1-t0):.1f}s")
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_map50", val_map, epoch)
        # checkpoint
        is_best = val_map > best_map
        state = {"epoch": epoch, "model_state": model.state_dict(), "optim_state": optimizer.state_dict(), "val_map": val_map}
        save_checkpoint(state, args.ckpt_dir, epoch, is_best=is_best)
        if is_best:
            best_map = val_map
            print(f"*** New best model (val mAP50 = {best_map:.4f}) saved.")
        # optional test each epoch
        if args.test_after_epoch:
            test_map, _ = evaluate(model, test_loader, device, anchors, args.num_classes)
            writer.add_scalar("epoch/test_map50", test_map, epoch)
            print(f"Test mAP50: {test_map:.4f}")
    writer.close()
    print("Training complete. Best val mAP50:", best_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./Datasets/Drive_India")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_size", type=int, default=640)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--step_step", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=24)
    parser.add_argument("--log_dir", type=str, default="./runs")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--test_after_epoch", action="store_true")
    args = parser.parse_args()
    main(args)
