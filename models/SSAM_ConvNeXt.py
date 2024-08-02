from torchvision.models import convnext_small, ConvNeXt_Small_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        x = nn.Flatten(start_dim=1)(x)
        return self.layers(x)

class SSAM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super(SSAM, self).__init__()
        self.spectral_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.channel_reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        sa = self.spectral_attention(x)
        x = x * sa
        spa = self.spatial_attention(x)
        x = x * spa
        x = self.channel_reduction(x)
        return x

class SharedSSAM_FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SharedSSAM_FPN, self).__init__()
        
        # 共享的SSAM
        self.ssam = SSAM(in_channels, out_channels)
        
        # FPN層
        self.lateral_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.lateral_conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.fpn_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.fpn_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.fpn_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.fpn_conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # 應用共享的SSAM到輸入，生成4個特徵圖
        c1 = self.ssam(x)
        c2 = self.ssam(F.max_pool2d(x, kernel_size=2, stride=2))
        c3 = self.ssam(F.max_pool2d(x, kernel_size=4, stride=4))
        c4 = self.ssam(F.max_pool2d(x, kernel_size=8, stride=8))

        # FPN自頂向下路徑和橫向連接
        p4 = self.lateral_conv4(c4)
        p3 = self._upsample_add(p4, self.lateral_conv3(c3))
        p2 = self._upsample_add(p3, self.lateral_conv2(c2))
        p1 = self._upsample_add(p2, self.lateral_conv1(c1))

        # FPN輸出卷積
        p4 = self.fpn_conv4(p4)
        p3 = self.fpn_conv3(p3)
        p2 = self.fpn_conv2(p2)
        p1 = self.fpn_conv1(p1)

        return p1, p2, p3, p4

class FeatureFusion(nn.Module):
    def __init__(self, in_channels, fusion_method='concat'):
        super(FeatureFusion, self).__init__()
        self.fusion_method = fusion_method
        
        self.fusion_conv = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1)

    def forward(self, features):
        # 上采样所有特征图到最大分辨率
        max_size = features[0].size()[2:]
        upsampled_features = [F.interpolate(f, size=max_size, mode='bilinear', align_corners=False) for f in features]
        fused = torch.cat(upsampled_features, dim=1)
        fused = self.fusion_conv(fused)
        return fused

class ConvNeXt_FPN_Classifier(nn.Module):
    def __init__(self, in_channels=101, out_channels=64, num_classes=2, mlp_hidden_channels=256):
        super(ConvNeXt_FPN_Classifier, self).__init__()
        self.ssam_fpn = SharedSSAM_FPN(in_channels, out_channels)
        self.fusioner = FeatureFusion(out_channels)
        self.convnext = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)

        self.convnext.features[0][0] = nn.Conv2d(out_channels, 96, kernel_size=(4, 4), stride=(4, 4))

        self.convnext.classifier = MLP(768, hidden_dim=mlp_hidden_channels, output_dim=num_classes)

        # 如果需要，可以凍結Swin Transformer的部分參數
        for param in self.convnext.parameters():
            param.requires_grad = False

        self.convnext.features[0][0].requires_grad_(True)
        self.convnext.classifier.requires_grad_(True)

    def forward(self, x):
        features = self.ssam_fpn(x)
        fusion_features = self.fusioner(features)
        
        output = self.convnext(fusion_features)
        return output