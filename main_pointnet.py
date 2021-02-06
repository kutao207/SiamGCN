import os.path as osp

import torch 
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool

from change_dataset import ChangeDataset, MyDataLoader
from transforms import NormalizeScale, SamplePoints

from pointnet2 import SAModule, GlobalSAModule, MLP


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        input_feature_dim = 6
        self.sa1_module = SAModule(0.5, 0.2, MLP([input_feature_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 10)

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

def train(epoch):
    model.train()
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
            loss = F.nll_loss(out, data.y)
            pred = out.max(1)[1]
            correct += pred.eq(data.y).sum().item()
        else:
            loss = F.nll_loss(model(data), data.y)

        loss.backward()
        optimizer.step()
    
    train_acc = correct / len(train_loader.dataset)
    print('Epoch: {:03d}, Train: {:.4f}'.format(epoch, train_acc))

def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

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
    path = 'F:/shrec2021/data'
    checkpoint_dir = 'checkpoints_main_pointnet'

    pre_transform, transform = NormalizeScale(), SamplePoints(1024)

    # train_dataset = ChangeDataset(path, train=True, clearance=3, transform=None, pre_transform=None)
    # train_dataset = ChangeDataset(path, train=True, clearance=3, transform=None, pre_transform=None)
    train_dataset = ChangeDataset(path, train=True, clearance=3, transform=transform, pre_transform=pre_transform)
    test_dataset = ChangeDataset(path, train=False, clearance=3, transform=transform, pre_transform=pre_transform)

    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)

    train_loader = MyDataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_loader = MyDataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    test_accs = []
    max_acc = 0
    for epoch in range(1, 201):

        train(epoch)
        test_acc = test(train_loader)
        if test_acc > max_acc:
            torch.save(model.state_dict(), 'best_pointnet_model.pth')
            max_acc = test_acc
        print('Epoch: {:03d}, Test: {:.4f}'.format(epoch, test_acc))

        # print("Breakpoint")




