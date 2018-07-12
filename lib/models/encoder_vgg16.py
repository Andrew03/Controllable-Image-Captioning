import torch.nn as nn
import torchvision.models as models

class EncoderVGG16(nn.Module):
    def __init__(self, is_normalized=False):
        super(EncoderVGG16, self).__init__()
        vgg = models.vgg16_bn(pretrained=True).eval() if is_normalized else models.vgg16(pretrained=True).eval()
        for param in vgg.parameters():
            param.requires_grad_(False)
        self.vgg = nn.Sequential(*(vgg.features[i] for i in range(29)))

    def forward(self, images):
        features = self.vgg(images)
        features_reshaped = features.view(images.size(0), 512, 196)
        features_transposed = features_reshaped.transpose(1, 2)
        return features_transposed
