import torch
import torch.nn as nn
from torchvision import models
import config

def get_model():
    """返回一个图像分类模型"""
    if config.model_name == 'resnet18':
        model = models.resnet18(pretrained=True)   
        
        model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    elif config.model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, config.num_classes)
    else:
        raise ValueError(f'不支持的模型: {config.model_name}')
    return model


if __name__ == '__main__':
    model = get_model()
    print(model)
    dummy = torch.randn(1, 3, 32, 32)
    out = model(dummy)
    print('输出形状:', out.shape)  # 应该是 [1, 10]