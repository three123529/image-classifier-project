import torch
data_root = '/home/aistudio/data'
checkpoint_dir =  './checkpoints' 
log_dir = './logs'             

batch_size = 64                
num_workers = 2               
num_epochs = 21                
learning_rate = 0.001          
momentum = 0.9                
weight_decay = 5e-4            

num_classes = 10               
model_name = 'resnet18'        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')