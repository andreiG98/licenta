import torch.nn as nn
import torch

class NetworkGradCam(nn.Module):
    def __init__(self, feature_extractor, num_classes, feature_extract):
        super().__init__()

        # gradient placeholder
        self.gradient = None

        if hasattr(feature_extractor, 'fc'): # resnets
            # isolate the feature blocks
            self.features = nn.Sequential(feature_extractor.conv1,
                                          feature_extractor.bn1,
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                          feature_extractor.layer1, 
                                          feature_extractor.layer2, 
                                          feature_extractor.layer3, 
                                          feature_extractor.layer4)
            # average pooling layer
            self.avgpool = feature_extractor.avgpool

            in_features = feature_extractor.fc.in_features
            if feature_extract:
                self.classifier = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=512),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(in_features=512, out_features=256),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(in_features=256, out_features=num_classes)
                )
            else:
                # classifier
                self.classifier = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        else: # vgg
            self.features = feature_extractor.features
            if feature_extract:
                self.features.eval()

            # average pooling layer
            self.avgpool = feature_extractor.avgpool

            in_features_first = feature_extractor.classifier[0].in_features
            in_features_last = feature_extractor.classifier[-1].in_features
            self.classifier = nn.Sequential(
                    nn.Linear(in_features=in_features_first, out_features=in_features_last, bias=True),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(in_features=in_features_last, out_features=in_features_last, bias=True),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(in_features=in_features_last, out_features=num_classes, bias=True)
                )
            
    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    def get_activations(self, x):
        return self.features(x)

    def forward(self, x):
        # extract the features
        x = self.features(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # complete the forward pass
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x