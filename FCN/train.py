import os 
from fcn32s import *
print(os.path.abspath('.'))
train_iter, val_iter = load_data_VOCSegmentation(year="2011", batch_size=8, crop_size=(320, 480),\
      root='Datasets/VOC/',num_workers=4, use=1)
torch.manual_seed(0)
import numpy as np
from torch.optim import lr_scheduler

net = FCN32s(num_classes=21)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=5e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
trainer(net, train_iter, val_iter, nn.CrossEntropyLoss(), \
optimizer, scheduler, num_epochs=100, gpu_id=None)