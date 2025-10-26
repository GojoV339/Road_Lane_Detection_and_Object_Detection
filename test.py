# test.py
import torch, argparse
from drive_dataloader import make_loaders
from drive_object_detection_model import DriveSingleShotModel
from utils import build_anchors_for_model, decode_outputs, compute_map50

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = make_loaders(args.data_root, batch_size=args.batch_size, input_size=args.input_size, num_workers=2)
    model = DriveSingleShotModel(num_classes=args.num_classes, input_size=args.input_size).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    anchors = build_anchors_for_model(model, input_size=args.input_size)
    # evaluate
    from utils import move_batch_to_device
    preds=[]; gts=[]
    model.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = move_batch_to_device((images, targets), device)
            outputs = model(images)
            batch_preds = decode_outputs(outputs, anchors, input_size=model.input_size, conf_thresh=0.05)
            preds.extend(batch_preds); gts.extend(targets)
    mAP50, per_class_ap = compute_map50(preds, gts, num_classes=args.num_classes)
    print("Test mAP50:", mAP50)
    for cls, ap in per_class_ap.items():
        if ap is not None:
            print(f"Class {cls}: AP {ap:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./Datasets/Drive_India")
    parser.add_argument("--ckpt_path", default="./checkpoints/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_size", type=int, default=640)
    parser.add_argument("--num_classes", type=int, default=24)
    args = parser.parse_args()
    main(args)
