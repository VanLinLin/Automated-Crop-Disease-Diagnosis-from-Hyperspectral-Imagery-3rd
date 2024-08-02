import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights

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
        return self.layers(x)

class SSAM_Swin_Classifier(nn.Module):
    def __init__(self, in_channels=101, out_channels=64, num_classes=2, mlp_hidden_dim=256):
        super(SSAM_Swin_Classifier, self).__init__()
        self.ssam = SSAM(in_channels, out_channels)
        
        # 加載預訓練的Swin Transformer模型
        self.swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        
        # 修改Swin Transformer的第一個卷積層以接受3通道輸入
        self.swin.features[0][0] = nn.Conv2d(out_channels, 96, kernel_size=(4, 4), stride=(4, 4))
        
        # 修改最後的分類層以輸出指定的類別數
        swin_out_features = self.swin.head.in_features
        self.swin.head = nn.Identity()


        self.mlp = MLP(swin_out_features, hidden_dim=mlp_hidden_dim, output_dim=num_classes)

        # 如果需要，可以凍結Swin Transformer的部分參數
        for param in self.swin.parameters():
            param.requires_grad = False

        # 只訓練SSAM、Swin的第一層卷積和最後的分類層
        for param in self.ssam.parameters():
            param.requires_grad = True

        self.swin.features[0][0].requires_grad_(True)
        self.swin.head.requires_grad_(True)


    def forward(self, x):
        x = self.ssam(x)
        x = self.swin(x)
        x = self.mlp(x)
        return x