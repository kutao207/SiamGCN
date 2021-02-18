import os
import os.path as osp

import torch
from main_pointnet import Net
from transforms import NormalizeScale, SamplePoints
from change_dataset import ChangeDataset, MyDataLoader


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

    pre_transform, transform = NormalizeScale(), SamplePoints(1024)

    # train_dataset = ChangeDataset(path, train=True, clearance=3, transform=transform, pre_transform=pre_transform)
    test_dataset = ChangeDataset(path, train=False, clearance=3, transform=transform, pre_transform=pre_transform)

    # train_loader = MyDataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = MyDataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    test_acc = inference(test_loader)

    print("Test accuracy: {:.5f}".format(test_acc))




