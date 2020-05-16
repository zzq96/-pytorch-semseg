import torch
import matplotlib.pyplot as plt
# from torchsummary import summary
import torchvision
from torch import nn
import matplotlib
import utils
import time
import PIL
from PIL import Image
import numpy as np
import torchvision.models as models
import os

def load_data_VOCSegmentation(year='2012', batch_size = 62, crop_size=None, root='../../Datasets/VOC', num_workers=4, use=1):

    print('year=%d, batch_size=%')
    voc_train = utils.VOCSegmentation(root=root, year=year, image_set='train', crop_size=crop_size, use=use)
    print('已读取train, 共有%s张图片'%len(voc_train))
    voc_val = utils.VOCSegmentation(root=root, year=year, image_set='val', crop_size=crop_size, use=use)
    print('已读取val, 共有%s张图片'%len(voc_val))
    train_iter = torch.utils.data.DataLoader(voc_train, batch_size=batch_size, shuffle=True, num_workers= num_workers)
    val_iter = torch.utils.data.DataLoader(voc_val, batch_size=batch_size, shuffle=True, num_workers= num_workers)
    return train_iter, val_iter
    
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """造一个双线性插值卷积核"""
    """通过测试"""
    factor = (kernel_size + 1)//2 # 采样因子
    if kernel_size % 2 == 1:
        center = factor - 1 #采样点
    else:
        center = factor - 0.5
    
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor) #根据像素点离采样点的远近分配权重
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)
    
class FCN32s(nn.Module):

    def __init__(self, num_classes):
        super(FCN32s, self).__init__()

        assert num_classes > 0
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512,4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096,4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32,
                                          bias=False) 
                                        
        self._initialize_weights()
        
        vgg16 = models.vgg16(pretrained=True)
        self.copy_params_from_vgg16(vgg16)
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                # nn.init.kaiming_normal_(m.weight.data)
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, X):
        h = X
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore(h)        
        h = h[:, :, 19:19 + X.size()[2], 19:19 + X.size()[3]].contiguous()

        return h

    def copy_params_from_vgg16(self,vgg16):
        features=[
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5
        ]
        for l1, l2 in zip(features, vgg16.features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.data.shape == l2.weight.data.shape
                assert l1.bias.data.shape ==  l2.bias.data.shape
                l1.weight.data.copy_(l2.weight.data)
                l1.bias.data.copy_(l2.bias.data)
        for name, i in zip(['fc6', 'fc7'], [0, 3]):
            l1 = getattr(self, name)
            l2 = vgg16.classifier[i]
            l1.weight.data.copy_(l2.weight.data.view(l1.weight.data.shape))
            l1.bias.data.copy_(l2.bias.data.view(l1.bias.data.shape))


def _fast_hist(label_true, label_pred, n_class):
    """
    Inputs:
    - label_true: numpy, (W, )
    = label_pred: numpy, (W, )

    Returns:
    - hist: numpy, (n_class, n_class), hist[i, j] shows the number of the true label i to pred label j
    """
    mask = (label_true < 255) & (label_true < n_class)
    x = n_class * label_true[mask].astype(int) + label_pred[mask]
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """
    Inputs:
    - label_trues: list, (H, w), numpy
    - label_preds: list, (H, w), numpy
    - n_class: int
    Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def evaluate_accuracy(data_iter ,net, lossf, device):
    """
    Inputs:
    - data_iter:
    - net:
    - lossf:
    - device:
    """
    assert isinstance(net, nn.Module)
    loss, acc, acc_cls, mean_iu, fwavacc = 0, 0, 0, 0, 0
    eval_acc, eval_acc_cls, eval_mean_iu, eval_fwavacc = 0, 0, 0, 0
    loss_sum, n = 0.0, 0
    cnt = 0
    for X, y in data_iter:
        cnt += 1
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        l = lossf(y_hat, y)
        loss += l.cpu().item()
        tmp = y_hat.max(dim=1)
        label_pred = tmp[1].data.cpu().numpy()
        label_true = y.data.cpu().numpy()
        for lbt, lbp in zip(label_true, label_pred):
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, y_hat.shape[1])
            eval_acc += acc
            eval_acc_cls += acc_cls
            eval_mean_iu += mean_iu
            eval_fwavacc += fwavacc
        n += y.shape[0]
    return loss / cnt, eval_acc / n, eval_acc_cls / n, eval_mean_iu / n, eval_fwavacc / n


def trainer(net, train_iter, val_iter, loss_f, optimizer, scheduler, num_epochs, gpu_id = 0):

    accumulation_steps = 1
    # gpu_id == None，说明使用cpu
    device = torch.device("cuda" if gpu_id != None else 'cpu')
    if gpu_id:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    net = net.to(device)
    print("training on", device)
    step_cnt = 0

    with torch.no_grad():
        net.eval()
        train_loss, train_acc, train_acc_cls, train_mean_iu, train_fwavacc = evaluate_accuracy(train_iter, net, loss_f, device)
        val_loss, val_acc, val_acc_cls,val_mean_iu, val_fwavacc = evaluate_accuracy(val_iter, net, loss_f, device)
        print("epoch: begin")
        print("train_loss: %f, train_acc: %f, train_acc_cls:%f, train_mean_iu:%f, train_fwavacc:%f" % (train_loss, train_acc, train_acc_cls, train_mean_iu, train_fwavacc))
        print("val_loss: %f, val_acc: %f, val_acc_cls:%f, val_mean_iu:%f, val_fwavacc:%f" % (val_loss, val_acc, val_acc_cls, val_mean_iu, val_fwavacc))

    for epoch in range(num_epochs):
        start_time = time.time()
        net.train()
        for X, labels in train_iter:
            step_cnt += 1

            X = X.to(device)
            labels = labels.to(device)
            scores = net(X)
            loss = loss_f(scores, labels)
            #print("loss",loss.cpu().item())
            loss = loss/accumulation_steps
            loss.backward()
            if step_cnt % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                #print("lr", optimizer.param_groups[0]['lr'])
        scheduler.step()
                

        with torch.no_grad():
            net.eval()
            train_loss, train_acc, train_acc_cls, train_mean_iu, train_fwavacc = evaluate_accuracy(train_iter, net, loss_f, device)
            val_loss, val_acc, val_acc_cls,val_mean_iu, val_fwavacc = evaluate_accuracy(val_iter, net, loss_f, device)
            print("epoch: %d, time: %d sec" % (epoch + 1, time.time() - start_time))
            print("lr", optimizer.param_groups[0]['lr'])
            print("train_loss: %f, train_acc: %f, train_acc_cls:%f, train_mean_iu:%f, train_fwavacc:%f" % (train_loss, train_acc, train_acc_cls, train_mean_iu, train_fwavacc))
            print("val_loss: %f, val_acc: %f, val_acc_cls:%f, val_mean_iu:%f, val_fwavacc:%f" % (val_loss, val_acc, val_acc_cls, val_mean_iu, val_fwavacc))