import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Current cuda device is', device)

batch_size = 50
learning_rate = 0.0001
epoch_num = 15

# LOAD DATASET
import lib.dataset as ds

transform = A.Compose(
    [
        A.Resize(28, 28),
        ToTensorV2(),
    ]
)

path = 'D:/dev/Luminoso Aula/luminoso-ai/input/riddikulus_ds'
riddik_ds = ds.RiddikulusDataset(path, device=device, transform=transform)
dataset_loader = torch.utils.data.DataLoader(dataset=riddik_ds, shuffle=True)

print('-----------------------------------------------')
print('Dataset size:', len(riddik_ds), 'images')
print('Classes:', len(riddik_ds.classes))
print('-----------------------------------------------')

model = torch.jit.load('model/MODEL_PYTORCH_SCRIPT.pth')
model.eval()

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print('-----------------------------------------------')

correct = 0
total = len(dataset_loader.dataset)

print("Evaluating model with", total, "images...", end="")

for data, target in dataset_loader:
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()
print('done!\nAccuracy : {:.2f}%'.format(100. * correct / len(dataset_loader.dataset)))
