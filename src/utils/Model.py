import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
class MobileNetV3(nn.Module):
    def __init__(self, num_classes=2):  # Change to 2 for binary classification
        super(MobileNetV3, self).__init__()
        self.pretrained_model = timm.create_model('mobilenetv3_large_100', pretrained=True)  # Load pretrained MobileNetV3 model
        self.pretrained_model.classifier = nn.Linear(self.pretrained_model.classifier.in_features, 512)  # Intermediate layer
        self.final_fc = nn.Linear(512, num_classes)  # Two outputs for binary classification
    def forward(self, x):
        x = self.pretrained_model(x)  # Get the output from the pretrained model
        x = F.relu(x)  # Apply ReLU activation
        x = self.final_fc(x)  # Output layer
        return x  # No activation applied here, as CrossEntropyLoss will handle softmax
def MobileNetV3Model(num_classes):
    model = MobileNetV3(num_classes=num_classes)
    return model

        