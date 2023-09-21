
from torch import nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator



class DetectionModel(nn.Module):

    def __init__(self, num_classes):
        super(DetectionModel, self).__init__()
        self.num_classes = num_classes

        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.backbone.out_channels = 2048

        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
        self.model = FasterRCNN(self.backbone,
                   num_classes=self.num_classes+1,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)


    def get_embeddings(self, logo_crops):
        return self.backbone(logo_crops).mean([2,3])


    def forward(self, x, targets=None, logo_crops=None):

        if targets is not None:
            embeddings = self.get_embeddings(logo_crops)
            return self.model(x, targets), embeddings
        else:
            return self.model(x)  