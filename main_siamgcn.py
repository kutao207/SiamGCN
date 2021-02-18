import os
import os.path as osp

from datetime import datetime
import argparse

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Dropout
from torch.optim.lr_scheduler import StepLR

import torch_geometric

from change_dataset import ChangeDataset, MyDataLoader
from transforms import NormalizeScale, SamplePoints
from metric import ConfusionMatrix
from imbalanced_sampler import ImbalancedDatasetSampler
from pointnet2 import MLP

from torch_geometric.nn import DynamicEdgeConv, global_max_pool

from utils import ktprint, set_logger, check_dirs

#### log file setting
print  = ktprint
cur_filename = osp.splitext(osp.basename(__file__))[0]
log_dir = 'logs'
check_dirs(log_dir)
log_filename = osp.join(log_dir, '{}_{date:%Y-%m-%d_%H_%M_%S}'.format(cur_filename, date=datetime.now())+'.logging')
set_logger(log_filename)

#### log file setting finished!
#     0           1       2         3         4
# ["nochange","removed","added","change","color_change"]

NUM_CLASS = 5
USING_IMBALANCE_SAMPLING = True

class Net(torch.nn.Module):
    def __init__(self, k=20, aggr='max') -> None:
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 6, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 256]), k, aggr)
        self.lin = MLP([256, 512])

        self.lin_1 = Lin(512, 128)
        self.lin_2 = Lin(512, 128)

        self.mlp = Seq(
            MLP([128, 64]), Dropout(0.5),
            Lin(64, NUM_CLASS))
    
    def forward(self, data):
        r""""
        Args:
            data: [x], BN x 6, point clouds of 2016
                  [x2], BN x 6, point clouds of 2020
                  [batch], BN1, batch index of point clouds in 2016
                  [batch2], BN2, batch index of point clouds in 2020
                  [y], B, label
        Returns:
            out: []， Bx[NUM_CLASS]
        """

        b1_input = (data.x[:, 3:], data.x[:,:3], data.batch) # (feature, position, batch)
        b2_input = (data.x2[:,3:], data.x2[:,:3], data.batch2)

        b1_out_1 = self.conv1(data.x, data.batch)
        b1_out_2 = self.conv2(b1_out_1, data.batch)


        b2_out_1 = self.conv1(data.x2, data.batch2)
        b2_out_2 = self.conv2(b2_out_1, data.batch2)

        b1_out = global_max_pool(b1_out_2, data.batch)
        b2_out = global_max_pool(b2_out_2, data.batch2)

        b1, b2 = self.lin(b1_out), self.lin(b2_out)

        x = b2 - b1

       
        x1, x2 = self.lin_1(x), self.lin_2(x)

        x_out = F.dropout(F.relu(x1-x2), p=0.6)

        x_out = self.mlp(x_out)

        return F.log_softmax(x_out, dim=-1)

class Net_2(torch.nn.Module):
    def __init__(self, k=20, aggr='max') -> None:
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 6, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 256]), k, aggr)
        # self.lin = MLP([256, 512])

        # self.lin_1 = Lin(512, 128)
        # self.lin_2 = Lin(512, 128)

        # self.mlp = Seq(
        #     MLP([128, 64]), Dropout(0.5),
        #     Lin(64, NUM_CLASS))
        self.mlp2 = Seq(
            MLP([256, 64]), Dropout(0.5),
            Lin(64, NUM_CLASS))
    
    def forward(self, data):
        r""""
        Args:
            data: [x], BN x 6, point clouds of 2016
                  [x2], BN x 6, point clouds of 2020
                  [batch], BN1, batch index of point clouds in 2016
                  [batch2], BN2, batch index of point clouds in 2020
                  [y], B, label
        Returns:
            out: []， Bx[NUM_CLASS]
        """        
        # A simplified network
        b1_out_1 = self.conv1(data.x, data.batch)
        b1_out_2 = self.conv2(b1_out_1, data.batch)


        b2_out_1 = self.conv1(data.x2, data.batch2)
        b2_out_2 = self.conv2(b2_out_1, data.batch2)

        b1_out = global_max_pool(b1_out_2, data.batch)
        b2_out = global_max_pool(b2_out_2, data.batch2)

        # b1, b2 = self.lin(b1_out), self.lin(b2_out)
        # x = b2 - b1       
        # x1, x2 = self.lin_1(x), self.lin_2(x)
        # x_out = F.dropout(F.relu(x1-x2), p=0.6)

        x_out = b2_out - b1_out

        x_out = self.mlp2(x_out)

        return F.log_softmax(x_out, dim=-1)


def train(epoch):
    model.train()

    confusion_matrix = ConfusionMatrix(NUM_CLASS + 1)

    correct = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()        

        out = model(data)
        loss = F.nll_loss(out, data.y)
        pred = out.max(1)[1]
        correct += pred.eq(data.y).sum().item()        

        loss.backward()
        optimizer.step()

        confusion_matrix.increment_from_list(data.y.cpu().detach().numpy() + 1, pred.cpu().detach().numpy() + 1)
    
    train_acc = correct / len(train_loader.dataset)
    print('Epoch: {:03d}, Train: {:.4f}, per_class_acc: {}'.format(epoch, train_acc, confusion_matrix.get_per_class_accuracy()))

def test(loader):
    model.eval()
    confusion_matrix = ConfusionMatrix(NUM_CLASS+1)
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()

        confusion_matrix.increment_from_list(data.y.cpu().detach().numpy() + 1, pred.cpu().detach().numpy() + 1)
    test_acc = correct / len(loader.dataset)

    print('Epoch: {:03d}, Test: {:.4f}, per_class_acc: {}'.format(epoch, test_acc, confusion_matrix.get_per_class_accuracy()))

    return test_acc, confusion_matrix.get_per_class_accuracy()

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4)
    
    return parser.parse_args()




if __name__ == '__main__':

    data_root_path = '/home/kt/cyclomedia_disk/kt/shrec2021/data'
    if os.name == 'nt':
        data_root_path = 'F:/shrec2021/data'

    ignore_labels = [] # ['nochange']

    my_args = get_args()
    print(my_args)

    pre_transform, transform = NormalizeScale(), SamplePoints(1024)

    train_dataset = ChangeDataset(data_root_path, train=True, clearance=3, ignore_labels=ignore_labels, transform=transform, pre_transform=pre_transform)
    test_dataset = ChangeDataset(data_root_path, train=False, clearance=3, ignore_labels=ignore_labels, transform=transform, pre_transform=pre_transform)

    NUM_CLASS = len(train_dataset.class_labels)

    sampler = ImbalancedDatasetSampler(train_dataset)

    if not USING_IMBALANCE_SAMPLING:
        train_loader = MyDataLoader(train_dataset, batch_size=my_args.batch_size, shuffle=True, num_workers=my_args.num_workers)
    else:
        train_loader = MyDataLoader(train_dataset, batch_size=my_args.batch_size, shuffle=False, num_workers=my_args.num_workers, sampler=sampler)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = Net().to(device)
    model = Net_2().to(device)


    print(f"Using model: {model.__class__.__name__}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    test_accs = []
    max_acc = 0
    epoch_best = 1
    for epoch in range(1, 201):
        train(epoch) # Train one epoch
        test_acc, per_cls_acc = test(train_loader) # Test
        scheduler.step() # Update learning rate
        if test_acc > max_acc:
            torch.save(model.state_dict(), f'best_gcn_model_{model.__class__.__name__}.pth')
            max_acc = test_acc
            epoch_best = epoch
    print('Epoch: {:03d}, get best acc: {:.4f}, per class acc: {}'.format(epoch_best, test_acc, per_cls_acc))
