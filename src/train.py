import torch
from torch import nn
from torch.utils import data
from torchvision import transforms, datasets
from pretrainedmodels import se_resnext50_32x4d
from sklearn.metrics import roc_auc_score

# Parameters
validation_split = 0.2
batch_size = 128
num_epochs = 20
lr = 0.0007
size = 196

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Dataloader
train_dir = '../data/train'

transform = transforms.Compose([
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
        transforms.RandomRotation((0,0)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
        transforms.RandomRotation((90,90)),
        transforms.RandomRotation((180,180)),
        transforms.RandomRotation((270,270)),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation((90,90)),
        ]),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation((270,270)),
        ])
    ]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(
    root=train_dir,
    transform=transform
)

split_len = [round((1-validation_split) * len(dataset)), round(validation_split * len(dataset))]
split = {x: y for x, y in zip(['train', 'val'], data.random_split(dataset, lengths=split_len))}

loader = {x: data.DataLoader(split[x], batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

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
    print("Number of GPUS:", torch.cuda.device_count())
    net = torch.nn.DataParallel(net)
net.to(device)

criterion = torch.nn.CrossEntropyLoss()
accuracy = lambda x, y: (torch.argmax(x, dim=1) == y).float().mean()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# Training
print('Epoch,Train_Loss,Validation_Loss,Train_Accuracy,Validation_Accuracy,Validation_AUC')
best_auc = 0
for epoch in range(num_epochs):

    # Train
    train_loss, train_acc = 0, 0
    net.train()
    if epoch > 1: scheduler.step()
    for batch_idx, (batch, labels) in enumerate(loader['train']):
        x, y = batch.to(device), labels.to(device)

        optimizer.zero_grad()

        out = net(x)
        loss = criterion(out, y)
        acc = accuracy(out, y)

        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        train_acc = acc.item()

    # Val
    val_loss, val_acc, val_auc= 0, 0, 0
    net.eval()
    with torch.no_grad():
        for batch_idx, (batch, labels) in enumerate(loader['val']):
            x, y = batch.to(device), labels.to(device)

            out = net(x)

            loss = criterion(out, y)
            acc = accuracy(out, y)
            auc = roc_auc_score(Tensor([[1, 0] if x.item() == 0 else [0, 1] for x in labels]).cpu(), out.cpu())

            val_loss += loss.item()
            val_acc += acc.item()
            val_auc += auc

        val_loss /= batch_idx
        val_acc /= batch_idx
        val_auc /= batch_idx

    # Save best model
    if val_auc > best_auc:
        torch.save(net.state_dict(), '../checkpoint/' + 'epp' + str(epoch + 1) + 'AUC' + str(val_auc)[:5] + '.pt')
        best_auc = val_auc

    print(epoch + 1, train_loss, val_loss, train_acc, val_acc, val_auc, optimizer.param_groups[0]['lr'], sep=",")