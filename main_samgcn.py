import os.path as osp

import torch
import torch.nn.functional as F

from change_dataset import ChangeDataset, MyDataLoader
from transforms import NormalizeScale, SamplePoints

class Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

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


if __name__ == '__main__':

    data_root_path = '/home/kt/cyclomedia_disk/kt/shrec2021/data'

    pre_transform, transform = NormalizeScale(), SamplePoints(1024)

    train_dataset = ChangeDataset(data_root_path, train=True, clearance=3, transform=transform, pre_transform=pre_transform)
    test_dataset = ChangeDataset(data_root_path, train=False, clearance=3, transform=transform, pre_transform=pre_transform)

    train_loader = MyDataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = MyDataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    test_accs = []
    max_acc = 0
    for epoch in range(1, 201):
        train(epoch)
        test_acc = test(train_loader)
        if test_acc > max_acc:
            torch.save(model.state_dict(), 'best_gcn_model.pth')
            max_acc = test_acc
        print('Epoch: {:03d}, Test: {:.4f}'.format(epoch, test_acc))
