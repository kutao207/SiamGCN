import os
import os.path as osp

import torch 
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch.optim.lr_scheduler import StepLR

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool

from change_dataset import ChangeDataset, MyDataLoader
from transforms import NormalizeScale, SamplePoints
from metric import ConfusionMatrix

from focal_loss import focal_loss
from contrastive_loss import ContrastiveLoss

from imbalanced_sampler import ImbalancedDatasetSampler

from pointnet2 import SAModule, GlobalSAModule, MLP

#     0           1       2         3         4
# ["nochange","removed","added","change","color_change"]

NUM_CLASS = 5
USING_FOCAL_LOSS = False
USING_IMBALANCE_SAMPLING = True
USING_CONTRASTIVE_LOSS = False

class Net_cas(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_feature_dim = 6
        self.sa1_module = SAModule(0.5, 0.2, MLP([input_feature_dim, 64, 64, 128]))
        self.sa1_global_module = GlobalSAModule(MLP([128+3, 256, 512, 1024]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))        
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024 * 2, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, NUM_CLASS)

    def forward(self, data):
        sa0_b1_input = (data.x[:,3:], data.x[:,:3], data.batch)
        sa0_b2_input = (data.x2[:,3:], data.x2[:,:3], data.batch2)

        sa1_b1_out = self.sa1_module(*sa0_b1_input)
        sa2_b1_out = self.sa2_module(*sa1_b1_out)
        sa3_b1_out = self.sa3_module(*sa2_b1_out)

        sa1_b1_global_out = self.sa1_global_module(*sa1_b1_out)

        sa1_b2_out = self.sa1_module(*sa0_b2_input)
        sa2_b2_out = self.sa2_module(*sa1_b2_out)
        sa3_b2_out = self.sa3_module(*sa2_b2_out)

        sa1_b2_global_out = self.sa1_global_module(*sa1_b2_out)

        # x1, pos1, _ = sa3_b1_out
        # x2, pos2, _ = sa3_b2_out

        x1 = torch.cat((sa1_b1_global_out[0], sa3_b1_out[0]), -1)
        x2 = torch.cat((sa1_b2_global_out[0], sa3_b2_out[0]), -1)

        x = x1 + x2

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        input_feature_dim = 6
        self.sa1_module = SAModule(0.5, 0.2, MLP([input_feature_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, NUM_CLASS)

    def forward(self, data):
        sa0_b1_input = (data.x[:,3:], data.x[:,:3], data.batch)
        sa0_b2_input = (data.x2[:,3:], data.x2[:,:3], data.batch2)

        sa1_b1_out = self.sa1_module(*sa0_b1_input)
        sa2_b1_out = self.sa2_module(*sa1_b1_out)
        sa3_b1_out = self.sa3_module(*sa2_b1_out)

        sa1_b2_out = self.sa1_module(*sa0_b2_input)
        sa2_b2_out = self.sa2_module(*sa1_b2_out)
        sa3_b2_out = self.sa3_module(*sa2_b2_out)

        x1, pos1, _ = sa3_b1_out
        x2, pos2, _ = sa3_b2_out

        x = x1 + x2

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)

class Net_con(torch.nn.Module):
    def __init__(self):
        super().__init__()
        input_feature_dim = 6
        self.sa1_module = SAModule(0.5, 0.2, MLP([input_feature_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        # self.lin1 = Lin(1024, 512)
        # self.lin2 = Lin(512, 256)
        # self.lin3 = Lin(256, NUM_CLASS)

        self.lin = Seq(
            Lin(1024, 512), ReLU(), Dropout(p=0.5),
            Lin(512, 256), ReLU(), Dropout(p=0.5)            
            )
        self.lin_1 = Lin(256, NUM_CLASS)
        self.lin_2 = Lin(256, NUM_CLASS)

    def forward(self, data):
        sa0_b1_input = (data.x[:,3:], data.x[:,:3], data.batch)
        sa0_b2_input = (data.x2[:,3:], data.x2[:,:3], data.batch2)

        sa1_b1_out = self.sa1_module(*sa0_b1_input)
        sa2_b1_out = self.sa2_module(*sa1_b1_out)
        sa3_b1_out = self.sa3_module(*sa2_b1_out)

        sa1_b2_out = self.sa1_module(*sa0_b2_input)
        sa2_b2_out = self.sa2_module(*sa1_b2_out)
        sa3_b2_out = self.sa3_module(*sa2_b2_out)

        x1, pos1, _ = sa3_b1_out
        x2, pos2, _ = sa3_b2_out

        # x = x1 + x2

        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin3(x)
        # return F.log_softmax(x, dim=-1)
        x1 = self.lin(x1)
        x2 = self.lin(x2)
        x1_out = F.log_softmax(self.lin_1(x1), dim=-1)
        x2_out = F.log_softmax(self.lin_2(x2), dim=-1)

        return (x1_out, x2_out)

class Net_2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        input_feature_dim = 6
        self.sa1_module = SAModule(0.5, 0.2, MLP([input_feature_dim, 64, 64, 128]))
        # self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        # self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))
        self.sa3_module = GlobalSAModule(MLP([128 + 3, 256, 512, 1024]))

        # self.lin1 = Lin(1024, 512)
        # self.lin2 = Lin(512, 256)
        # self.lin3 = Lin(256, NUM_CLASS)

        # self.lin = Seq(
        #     Lin(1024, 512), ReLU(), Dropout(p=0.5),
        #     Lin(512, 256), ReLU(), Dropout(p=0.5)            
        #     )
        # self.lin_1 = Lin(256, NUM_CLASS)
        # self.lin_2 = Lin(256, NUM_CLASS)

        self.lin = Seq(
            Lin(1024, 256), ReLU(), Dropout(p=0.5)         
            )
        self.lin_1 = Lin(256, 128)
        self.lin_2 = Lin(256, 128)

        self.lin_last = Lin(128, NUM_CLASS)

    def forward(self, data):
        sa0_b1_input = (data.x[:,3:], data.x[:,:3], data.batch)
        sa0_b2_input = (data.x2[:,3:], data.x2[:,:3], data.batch2)

        # Using less layers

        # sa1_b1_out = self.sa1_module(*sa0_b1_input)
        # sa2_b1_out = self.sa2_module(*sa1_b1_out)
        # sa3_b1_out = self.sa3_module(*sa2_b1_out)

        # sa1_b2_out = self.sa1_module(*sa0_b2_input)
        # sa2_b2_out = self.sa2_module(*sa1_b2_out)
        # sa3_b2_out = self.sa3_module(*sa2_b2_out)

        sa1_b1_out = self.sa1_module(*sa0_b1_input)        
        sa3_b1_out = self.sa3_module(*sa1_b1_out)

        sa1_b2_out = self.sa1_module(*sa0_b2_input)        
        sa3_b2_out = self.sa3_module(*sa1_b2_out)

        x1, pos1, _ = sa3_b1_out
        x2, pos2, _ = sa3_b2_out

        x = x1 - x2

        x = self.lin(x)
        x1, x2 = self.lin_1(x), self.lin_2(x)
        x_out = F.dropout(F.relu(x1-x2), p=0.6)

        x_out = self.lin_last(x_out)

        return F.log_softmax(x, dim=-1)

        # x = x1 + x2

        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin3(x)
        # return F.log_softmax(x, dim=-1)

        # x1 = self.lin(x1)
        # x2 = self.lin(x2)
        # x1_out = F.log_softmax(self.lin_1(x1), dim=-1)
        # x2_out = F.log_softmax(self.lin_2(x2), dim=-1)

        # return (x1_out, x2_out)


def train(epoch):
    model.train()
    confusion_matrix = ConfusionMatrix(NUM_CLASS+1)
   

    if True:
        correct = 0
        # i = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        if True:
            # print(f"Iter {i}")
            # i += 1
            out = model(data)

            # if USING_FOCAL_LOSS:
            #     loss = focal_loss(out, data.y, alpha=0.5, reduction='mean')
            if USING_CONTRASTIVE_LOSS:
                loss = criterion(out[0], out[1], data.y)
            else:
                loss = F.nll_loss(out, data.y)
            pred = out.max(1)[1]
            correct += pred.eq(data.y).sum().item()
            
            confusion_matrix.increment_from_list(data.y.cpu().detach().numpy() + 1, pred.cpu().detach().numpy() + 1)
        else:
            if USING_FOCAL_LOSS:
                loss = focal_loss(model(data), data.y, alpha=0.5, reduction='mean')
            else:
                loss = F.nll_loss(model(data), data.y)
            
        loss.backward()
        optimizer.step()
    
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
    return test_acc

def inference(loader, path='best_pointnet_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)

    model.load_state_dict(torch.load(path))
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':
    
    path = '/home/kt/cyclomedia_disk/kt/shrec2021/data'

    if os.name == 'nt':
        path = 'F:/shrec2021/data'  

    checkpoint_dir = 'checkpoints_main_pointnet'

    ignore_labels = ['nochange']

    if USING_FOCAL_LOSS:
        print("Using focal loss!")
    if USING_IMBALANCE_SAMPLING:
        print("Using imbalance over sampling!")
    if USING_CONTRASTIVE_LOSS:
        print("Using contrastive loss!")

    pre_transform, transform = NormalizeScale(), SamplePoints(1024)
    

    # train_dataset = ChangeDataset(path, train=True, clearance=3, transform=None, pre_transform=None)
    # train_dataset = ChangeDataset(path, train=True, clearance=3, transform=None, pre_transform=None)
    train_dataset = ChangeDataset(path, train=True, clearance=3, ignore_labels=ignore_labels, transform=transform, pre_transform=pre_transform)
    test_dataset = ChangeDataset(path, train=False, clearance=3, ignore_labels=ignore_labels, transform=transform, pre_transform=pre_transform)

    NUM_CLASS = len(train_dataset.class_labels)

    sampler = ImbalancedDatasetSampler(train_dataset)

    if USING_CONTRASTIVE_LOSS:
        criterion = ContrastiveLoss()

    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)

    if not USING_IMBALANCE_SAMPLING:
        train_loader = MyDataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    else:
        train_loader = MyDataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4, sampler=sampler)
        
    test_loader = MyDataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if USING_CONTRASTIVE_LOSS:
        model = Net_con().to(device)
    else:
        model = Net_2().to(device)
        print("Using Net -> Net_2()")
    # model = Net_cas().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    test_accs = []
    max_acc = 0
    for epoch in range(1, 201):
        train(epoch)
        test_acc = test(train_loader)

        scheduler.step()

        if test_acc > max_acc:
            torch.save(model.state_dict(), 'best_pointnet_model_net2.pth')
            max_acc = test_acc
        

        # print("Breakpoint")




