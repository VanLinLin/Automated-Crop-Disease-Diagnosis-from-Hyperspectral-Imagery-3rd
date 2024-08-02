import torch.nn as nn


class EnsembleClassifier(nn.Module):
    def __init__(self, 
                 convnext_fpn_cls,
                 ssam_swin_cls,
                 rcan_cls) -> None:
        super().__init__()
        self.convnext_fpn_cls = convnext_fpn_cls
        self.ssam_swin_cls = ssam_swin_cls
        self.rcan_cls = rcan_cls

    def forward(self, img_32, img_64):
        rcan_probability = self.rcan_cls(img_32).softmax(dim=1)
        ssam_swin_probability = self.ssam_swin_cls(img_64).softmax(dim=1)
        convnext_fpn_probability = self.convnext_fpn_cls(img_64).softmax(dim=1)

        sum_probability = (rcan_probability + ssam_swin_probability + convnext_fpn_probability) / 3

        return sum_probability.argmax(dim=1)