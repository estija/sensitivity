import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
import matplotlib.pyplot as plt
import time
from vit_small import ViT
from cnn import ResNet18
from densenet import densenet121
from convmixer import ConvMixer
from simple_vit import SimpleViT
import argparse
import wandb
import random
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR

#Fix seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='Your description here')

# Add parameters with default values
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--sched', default=False, action='store_true', help='use LR Scheduler')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--imsize', type=int, default=32, help='Image size')
parser.add_argument('--num_im', type=int, default=50000, help='Number of images')
parser.add_argument('--bs', type=int, default=100, help='Batch size')
parser.add_argument('--opt', type=str, default='adam', help='optimizer')
parser.add_argument('--model', type=str, default='vit', help='model architecture to use')
parser.add_argument('--sfmax', action='store_false', help='Use softmax attention')
parser.add_argument('--depth', type=int, default=8, help='Depth')
parser.add_argument('--heads', type=int, default=32, help='Number of attention heads')
parser.add_argument('--dim', type=int, default=64, help='Dim')
parser.add_argument('--mlp_dim', type=int, default=512, help='MLP-Dim')
parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
parser.add_argument('--var', type=float, default=0.1, help='Variance for perturbation for sensitivity')
parser.add_argument('--num_reps', type=int, default=5, help='number of repetitions for computing sensitivity')
parser.add_argument('--reg', action='store_true', help='train with sensitivity regularization')
parser.add_argument('--noise', type=float, default=1, help='Variance for perturbation for sensitivity reg')
parser.add_argument('--p', type=float, default=0.5, help='regularization strength')
parser.add_argument('--shp', action='store_true', help='log sharpness values')
parser.add_argument('--corr', action='store_true', help='evaluate on corrupted test sets')
parser.add_argument('--corr_path', type=str, default='CIFAR-10-C/', help='path to directory containing corrupted test sets')
parser.add_argument('--sev', type=int, default=0, help='severity level while evaluating on corrupted set')

args = parser.parse_args()

if args.corr:
    metric_names = ["train acc", "train loss", "test acc", "test loss", "train sensitivity", "sharpness1", "sharpness2" "gaussian_noise", "impulse_noise", "shot_noise", "gaussian_blur", 
               "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"]
    dir_name = args.corr_path
else: 
    metric_names = ["train acc", "train loss", "test acc", "test loss", "train sensitivity", "sharpness1", "sharpness2"]

eval_epoch = 5
lr = args.lr
sched = args.sched
num_epochs = args.num_epochs
imsize = args.imsize
num_im = args.num_im
bs = args.bs
opt = args.opt
model_name = args.model
sfmax = args.sfmax
depth = args.depth
heads = args.heads
dim = args.dim
mlp_dim = args.mlp_dim
patch_size = args.patch_size
var = args.var
num_reps = args.num_reps
reg = args.reg
noise = args.noise
reg_st = args.p
sev = args.sev

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 dataset and dataloaders
transform = transforms.Compose([
    transforms.Resize((imsize,imsize)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((imsize,imsize)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
subset_indices = torch.randperm(len(train_dataset))[:num_im]
subset_sampler = SubsetRandomSampler(subset_indices)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=bs, sampler=subset_sampler)
test_loader = DataLoader(test_dataset, batch_size=bs)

#wandb
wandb.init(
        # set the wandb project where this run will be logged
        project="cifar-sens",    
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "epochs": num_epochs,
        "imsize": imsize,
        "bs": bs,
        "opt": opt,
        "model_name": model_name,
        "softmax": sfmax,
        "depth": depth,
        "heads": heads,
        "dim": dim,
        "mlp_dim": mlp_dim,
        "patch_size": patch_size,
        "var": var,
        "num_reps": num_reps,
        "reg": reg,
        "reg_st": reg_st,
        "noise": noise,
        "severity": sev,
        }
    )

# Useful functions
def get_model(model_name, patch_size, sfmax, dim, depth, heads, mlp_dim, device):
    if model_name == 'vit':
        model = ViT(image_size = imsize,
        patch_size = patch_size,
        num_classes = 10,
        sfmax = sfmax,
        dim = dim,
        depth = depth,
        heads = heads,
        mlp_dim = mlp_dim,
        dropout = 0.1,
        emb_dropout = 0.1).to(device)
    elif model_name == 'svit':
        model = SimpleViT(image_size = imsize,
        patch_size = patch_size,
        num_classes = 10,
        dim = dim,
        depth = depth,
        heads = heads,
        mlp_dim = mlp_dim).to(device)
    elif model_name == 'cnn':
        model = ResNet18().to(device)
    elif model_name == 'cnnd':
        model = densenet121().to(device)
    elif model_name == 'convm':
        model = ConvMixer(dim, depth, kernel_size=3, patch_size=patch_size, n_classes=10).to(device)

    return model

def compute_sensitivity(model, inputs, predicted, var, num_reps, device, imsize, patchsize):
    sens = 0
    batch_size = inputs.size()[0]
    imsize = inputs.size()[2]
    num_pixels = patchsize**2

    for i in range(num_reps):
        inputsi = inputs.clone()

        mask = torch.zeros(batch_size, 1, imsize, imsize).to(device)
        mask[:, :, torch.randint(0, imsize, (num_pixels,)), torch.randint(0, imsize, (num_pixels,))] = 1
        mask = mask.expand(batch_size, 3, imsize, imsize)
    
        inputsi += (var * torch.randn_like(inputs) * mask).to(device)
        
        _, preds = torch.max(model(inputsi).data, 1)
        sens += (predicted != preds).sum().item()
    return sens/num_reps

def test_model(model, test_loader, device, var, num_reps, imsize, patch_size, sens_flag):
    model.eval()

    running_loss = 0.0
    acc = 0
    sens = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            acc += (predicted == labels).sum().item()
            if sens_flag:
                sens += compute_sensitivity(model, inputs, predicted, var, num_reps, device, imsize, patch_size)

        te_loss = running_loss / len(test_loader)
        te_acc = acc / len(test_loader)
        te_sens = sens / len(test_loader)
    return te_loss, te_acc, te_sens

def get_loaders(metric_names, dir_name, transform, labels, sev):
    loaders = []
    idx = int(sev*10000)
    for name in metric_names[7:]:
        path_name = dir_name+name+'.npy'
        imgs = np.load(path_name)
        new_imgs = torch.stack([transform(img) for img in imgs[idx:idx+10000]])
        dl = DataLoader(TensorDataset(new_imgs, labels), batch_size=1000, shuffle=False)
        loaders.append(dl)
    return loaders

def perturb_model(model, model_name, patch_size, sfmax, dim, depth, heads, mlp_dim, device, std):
    model2 = get_model(model_name, patch_size, sfmax, dim, depth, heads, mlp_dim, device)

    for param1, param2 in zip(model.parameters(), model2.parameters()):
      param2.data.copy_(param1.data)
      noisep = torch.randn_like(param2) * std
      param2.data.add_(noisep)
        
    return model2

def check_sharpness(model, train_loader, device, bs, num_reps, model_name, patch_size, sfmax, dim, depth, heads, mlp_dim, std=0.05):
    sharpness1 = 0
    sharpness2 = 0
    total = 0
    #print(inputs.size())
    with torch.no_grad():
        for j in range(num_reps):
            model2 = perturb_model(model, model_name, patch_size, sfmax, dim, depth, heads, mlp_dim, device, std)
            for i, (inputs, labels) in enumerate(train_loader, 0):
                total += bs
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs2 = model2(inputs)
                sharpness1 += torch.abs(outputs - outputs2).sum().item()

                _, preds = torch.max(outputs.data, 1)
                _, preds2 = torch.max(outputs2.data, 1)
                sharpness2 += (preds!=preds2).sum().item()

    return sharpness1/total, sharpness2/total

def test_model_corruptions(model, test_loader, device):
    model.eval()

    #running_loss = 0.0
    acc = 0
    #sens = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            acc += (predicted == labels).sum().item()

        te_acc = acc / len(test_loader)
    return te_acc/10000


# Setup
model = get_model(model_name, patch_size, sfmax, dim, depth, heads, mlp_dim, device)
criterion = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()

if opt=='sgd':
  optimizer = optim.SGD(model.parameters(), lr=lr)
elif opt=='adam':
  optimizer = optim.Adam(model.parameters(), lr=lr)

scheduler = StepLR(optimizer, step_size=8, gamma=0.5)

if args.corr:   
    labels2 = np.load(dir_name+'labels.npy')
    labels2 = torch.from_numpy(labels2[int(sev*10000):int(sev*10000)+10000])
    corrupted_loaders = get_loaders(metric_names, dir_name, transform2, labels2, sev)


# Training loop
for epoch in range(num_epochs):
    #TEST
    if args.corr:
        metrics = torch.zeros(7+len(corrupted_loaders))
    else:
        metrics = torch.zeros(7)
    if epoch%eval_epoch==0:
        tr_loss, tr_acc, tr_sens = test_model(model, train_loader, device, var, num_reps, imsize, patch_size, True)
        te_loss, te_acc, _ = test_model(model, test_loader, device, var, num_reps, imsize, patch_size, False)
        metrics[0], metrics[1], metrics[2], metrics[3], metrics[4] = tr_acc/bs, tr_loss, te_acc/bs, te_loss, tr_sens/bs
        if args.shp:
            metrics[5], metrics[6] = check_sharpness(model, train_loader, device, bs, num_reps, model_name, patch_size, sfmax, dim, depth, heads, mlp_dim)
        else:
            metrics[5], metrics[6] = 0, 0
        if args.corr:
            j = 7
            for loader in corrupted_loaders:
                metrics[j] = test_model_corruptions(model, loader, device)
                j+=1
       
    #TRAIN
    model.train()
    
    running_loss = 0.0
    acc = 0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss1 = criterion(outputs, labels)
        
        if reg:
            _, preds = torch.max(outputs.data, 1)
            inputs1 = inputs.clone()
            for k in range(inputs.size()[0]):
                xstart = np.random.randint(0,imsize-patch_size)
                ystart = np.random.randint(0,imsize-patch_size)
                inputs1[k,:,xstart:xstart+patch_size,ystart:ystart+patch_size] += noise*torch.randn((3,patch_size,patch_size)).to(device)
            outputs1 = model(inputs1)
            loss2 = criterion2(outputs1, outputs) 
            ((1-reg_st)*loss1 + reg_st*loss2).backward()
        else:
            loss1.backward()
 
    if opt == 'adam' or sched:
        scheduler.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {tr_loss:.4f}")
    
    if epoch%eval_epoch==0:
        data_dict = dict(zip(metric_names, metrics))
        #LOG RESULTS
        wandb.log(data_dict, step=epoch//5) 

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
