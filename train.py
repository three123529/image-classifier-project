import torch
import torch.optim as optim
from tqdm import tqdm
import config
from data_loader import get_dataloader
from model import get_model
from loss import get_loss_function
from utils import setup_logger, save_checkpoint, load_checkpoint

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    logger = setup_logger()
    logger.info('开始训练...')
    logger.info(f'配置信息: {config.__dict__}')
    
    device = config.device
    logger.info(f'使用设备: {device}')
    
    train_loader = get_dataloader(train=True)
    val_loader = get_dataloader(train=False)
    logger.info(f'训练集批次: {len(train_loader)}, 验证集批次: {len(val_loader)}')
    
    model = get_model().to(device)
    logger.info(f'模型: {config.model_name}')
    
    criterion = get_loss_function()
    
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate,
                          momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    start_epoch, best_acc = load_checkpoint(model, optimizer, filename='last_checkpoint.pth.tar')
    logger.info(f'起始epoch: {start_epoch}, 当前最佳准确率: {best_acc:.2f}%')
    
    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f'Epoch {epoch+1}/{config.num_epochs}')
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, filename='best_model.pth.tar')
            logger.info(f'发现新的最佳模型，准确率 {best_acc:.2f}%，已保存')
        
        if (epoch + 1) % 5 == 0:  
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, filename='last_checkpoint.pth.tar')
    
    logger.info('训练结束！')

if __name__ == '__main__':
    main()