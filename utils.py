import os
import torch
import logging
from datetime import datetime
import config

def setup_logger():
    """设置日志记录器，同时输出到文件和控制台"""
    os.makedirs(config.log_dir, exist_ok=True)
    log_file = os.path.join(config.log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """保存模型权重"""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    filepath = os.path.join(config.checkpoint_dir, filename)
    torch.save(state, filepath)
    logging.info(f'模型已保存至 {filepath}')

def load_checkpoint(model, optimizer=None, filename='best_model.pth.tar'):
    """加载模型权重（用于恢复训练或验证）"""
    filepath = os.path.join(config.checkpoint_dir, filename)
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('best_acc', 0.0)
        logging.info(f'已加载权重 {filepath} (epoch {epoch})')
        return epoch, best_acc
    else:
        logging.warning(f'未找到权重文件 {filepath}')
        return 0, 0.0