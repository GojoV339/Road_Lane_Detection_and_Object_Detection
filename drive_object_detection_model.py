import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k = 3, s = 1, p = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias = False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace = True)
        
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))
    
    
class ResidualBlock(nn.Module):
    """
    Residual block that correctly handles channel changes and stride.
    If in_c != out_c or stride != 1, a downsample path is created to match dims.
    """
    def __init__(self, in_c, out_c=None, stride=1):
        super().__init__()
        if out_c is None:
            out_c = in_c
        
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out
    
class SmallResNet(nn.Module):
    """
    Produces feature maps c3, c4, c5 with channel dims:
      c3 -> base (e.g. 32)
      c4 -> base*2 (e.g. 64)
      c5 -> base*4 (e.g. 128)
    Spatial strides (approx): c3 -> /4, c4 -> /8, c5 -> /16
    """
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        # Stem: two 3x3 convs (first with stride=2) + optional maxpool to reach /4
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False), # /2
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), # stays /2
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # /2 -> total /4
        )

        # c3: keep base_channels
        self.layer1 = nn.Sequential(
            ResidualBlock(base_channels, base_channels, stride=1),
            ResidualBlock(base_channels, base_channels, stride=1)
        )
        # c4: increase channels to base*2 and downsample (stride=2)
        self.layer2 = nn.Sequential(
            ResidualBlock(base_channels, base_channels*2, stride=2),
            ResidualBlock(base_channels*2, base_channels*2, stride=1)
        )
        # c5: increase channels to base*4 and downsample again
        self.layer3 = nn.Sequential(
            ResidualBlock(base_channels*2, base_channels*4, stride=2),
            ResidualBlock(base_channels*4, base_channels*4, stride=1)
        )

    def forward(self, x):
        x = self.stem(x)        # spatial /4
        c3 = self.layer1(x)     # channels = base_channels
        c4 = self.layer2(c3)    # channels = base_channels*2
        c5 = self.layer3(c4)    # channels = base_channels*4
        return c3, c4, c5

    
class FPN(nn.Module):
    """
    Top-down fusion: merges c3,c4,c5 into multi-scale pyramid (p3,p4,p5).
    Outputs all maps with out_channels channels.
    """
    
    def __init__(self, in_channels_list = [32,64,128], out_channels = 256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(ic, out_channels, 1) for ic in in_channels_list])
        self.smooth = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, 1, 1) for _ in in_channels_list])
        
    def forward(self,feats):
        # feats = [c3, c4, c5] (low -> high)
        p5 = self.lateral[2](feats[2])
        p4 = self.lateral[1](feats[1]) + F.interpolate(p5, scale_factor=2, mode = 'nearest')
        p3 = self.lateral[0](feats[0]) + F.interpolate(p4, scale_factor=2, mode = 'nearest')
        p5 = self.smooth[2](p5)
        p4 = self.smooth[1](p4)
        p3 = self.smooth[0](p3)
        return [p3, p4, p5]
    
    
class DetectionHead(nn.Module):
    """
    Small head per FPN level that predicts:
        - classification logits (A * num_classes)
        - bbox regression ( A * 4)
        - objections / centerness ( A * 1)
    whe A = num_anchors_per_cell (we use 3).
    """
    def __init__(self, in_channels = 256, num_anchors = 2, num_classes = 24):
        super().__init__()
        mid = in_channels
        self.shared = nn.Sequential(
            ConvBNReLU(in_channels, mid, 3,1,1),
            ConvBNReLU(mid, mid, 3, 1, 1)
        )
        self.cls_conv = nn.Conv2d(mid, num_anchors * num_classes, 1)
        self.reg_conv = nn.Conv2d(mid, num_anchors * 4, 1)
        self.obj_conv = nn.Conv2d(mid, num_anchors * 1, 1)
        
    def forward(self,x):
        t = self.shared(x)
        cls = self.cls_conv(t) # predicts the classes
        reg = self.reg_conv(t) # predict the bounding boxes
        obj = self.obj_conv(t) # predicts objectness 
        return cls, reg, obj 
    
    
class DriveSingleShotModel(nn.Module):
    def __init__(self, num_classes = 24, input_size = 640, num_anchors = 3, base_scales = [0.04, 0.12, 0.36]):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.num_anchors = num_anchors
        self.base_scales = base_scales
        self.backbone = SmallResNet(base_channels=32)
        self.fpn = FPN([32, 64, 128], out_channels= 256)
        self.heads = nn.ModuleList([DetectionHead(256, num_anchors, num_classes) for _ in range(3)])
        
    def forward(self,x):
        c3,c4,c5 = self.backbone(x)
        pyramid = self.fpn([c3,c4,c5])
        out = [head(p) for head, p in zip(self.heads, pyramid)]
        return out
    
    def feature_map_sizes(self, input_shape):
        """Return list of (H,W) for p3,p4,p5 for a given input_shape (H,W)."""
        device = next(self.parameters()).device
        with torch.no_grad():
            dummy = torch.zeros(1,3, input_shape[0], input_shape[1], device = device)
            c3, c4, c5 = self.backbone(dummy)
            return [tuple(c.shape[-2:]) for c in (c3,c4,c5)]
        
    def build_anchors(self, input_size = None, device = None, ratios = [0.5,1.0,2.0]):
        """
        Returns anchors as tensor (A_total, 4) in normalized cx,cy,w,h (0..1),
        where A_total = sum(H*W*num_anchors) across pyramid levels.
        """
        
        if input_size is None: input_size = self.input_size
        if device is None: device = next(self.parameters()).device
        sizes = self.feature_map_sizes((input_size, input_size))
        anchors = []
        for (H,W), base_scale in zip(sizes, self.base_scales):
            for i in range(H):
                for j in range(W):
                    cy = (i + 0.5) / H
                    cx = (j + 0.5) / W
                    for r in ratios[:self.num_anchors]:
                        area = base_scale * base_scale
                        w = math.sqrt(area * r)
                        h = math.sqrt(area / r)
                        anchors.append([cx,cy,w,h])
        return torch.tensor(anchors, dtype = torch.float32, device = device)
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallResNet(in_channels=3, base_channels=32).to(device)
    dummy = torch.randn(2, 3, 640, 640, device=device)
    c3, c4, c5 = model(dummy)
    print("c3:", c3.shape)   # expect (2,32,H/4,W/4)
    print("c4:", c4.shape)   # expect (2,64,H/8,W/8)
    print("c5:", c5.shape)   # expect (2,128,H/16,W/16)
        