import torch
import config
from data_loader import get_dataloader
from model import get_model
from loss import get_loss_function
from utils import load_checkpoint

def main():
    device = config.device
    
    val_loader = get_dataloader(train=False)
    
    model = get_model().to(device)
    
    start_epoch, best_acc = load_checkpoint(model, filename='best_model.pth.tar')
    print(f'已加载最佳模型，之前验证准确率: {best_acc:.2f}%')
    
    criterion = get_loss_function()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f'本次验证准确率: {acc:.2f}%')

if __name__ == '__main__':
    main()