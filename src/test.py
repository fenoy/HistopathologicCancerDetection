import torch
from torch import nn
from torch.utils import data
from torchvision import transforms, datasets
from pretrainedmodels import se_resnext50_32x4d
import sys

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Dataloader
test_dir = '../data/test'
size = 196

transform = transforms.Lambda(
    lambda img: torch.stack([
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomChoice([
                transforms.ColorJitter(brightness=0.5),
                transforms.ColorJitter(contrast=0.5),
                transforms.ColorJitter(saturation=0.5),
                transforms.ColorJitter(hue=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            ]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])(img),
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomChoice([
                transforms.RandomRotation((0, 0)),
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270)),
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.RandomRotation((90, 90)),
                ]),
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.RandomRotation((270, 270)),
                ])
            ]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])(img),
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomChoice([
                transforms.ColorJitter(brightness=0.5),
                transforms.ColorJitter(contrast=0.5),
                transforms.ColorJitter(saturation=0.5),
                transforms.ColorJitter(hue=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            ]),
            transforms.RandomChoice([
                transforms.RandomRotation((0, 0)),
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270)),
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.RandomRotation((90, 90)),
                ]),
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.RandomRotation((270, 270)),
                ])
            ]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])(img),
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])(img)
    ])
)

dataset = datasets.ImageFolder(
    root=test_dir,
    transform=transform
)

idx_to_label = {str(value): key for key, value in dataset.class_to_idx.items()}

loader = data.DataLoader(dataset, 1, shuffle=False, num_workers=4)

# Model
net = se_resnext50_32x4d()
net.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
net.last_linear = nn.Sequential(
    nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.Dropout(p=0.8),
    nn.Linear(2048, 512, bias=True),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.Dropout(p=0.8),
    nn.Linear(512, 256, bias=True),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.Dropout(p=0.8),
    nn.Linear(256, 2)
)

if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)
net.to(device)

d = torch.load(sys.argv[1])
net.load_state_dict(d)

f = open("../submission.csv","w+")
f.write("id,label\n")

with torch.no_grad():
    for batch_idx, (batch, id) in enumerate(loader):
        net.eval()

        x = batch.to(device)

        batch_size, n_crops, c, h, w = x.size()
        x = x.view(-1, c, h, w)

        out = net(x)
        out = out.view(batch_size, n_crops, -1).mean(1)
        out = torch.argmax(out, dim=1).item()

        f.write(idx_to_label[str(id.item())] + "," + str(out) + "\n")