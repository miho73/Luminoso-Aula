import sys

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import DataLoader

from lib.model import CNN

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

running_path = sys.argv[0]
path = 'D:/dev/Luminoso Aula/luminoso-ai/input/riddikulus_ds'
riddik_ds = ds.RiddikulusDataset(path, device=device, transform=transform)

print('-----------------------------------------------')
print('Dataset size:', len(riddik_ds), 'images')
print('Classes:', len(riddik_ds.classes))
print('-----------------------------------------------')

batch_size = 50
nr_classes = len(riddik_ds.classes)

# SPLIT DATASET
print('Splitting dataset...', end='')
train_data, test_data = train_test_split(riddik_ds, test_size=0.2, random_state=42)

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size, shuffle=True)
first_batch = train_loader.__iter__().__next__()
print('done!')
print('-----------------------------------------------')
print('{:15s} | {:<25s} | {}'.format('name', 'type', 'size'))
print('{:15s} | {:<25s} | {}'.format('Num of Batch', '', len(train_loader)))
print('{:15s} | {:<25s} | {}'.format('first_batch', str(type(first_batch)), len(first_batch)))
print('{:15s} | {:<25s} | {}'.format('first_batch[0]', str(type(first_batch[0])), first_batch[0].shape))
print('{:15s} | {:<25s} | {}'.format('first_batch[1]', str(type(first_batch[1])), first_batch[1].shape))
print('-----------------------------------------------')

# TRAIN MODEL
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

model.train()
i = 1
for epoch in range(epoch_num):
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Epoch : {}, Batch : {}, Loss : {:3f}".format(epoch + 1, i, loss.item()))
        i += 1
    i = 1

# EVALUATE MODEL
model.eval()  # 평가시에는 dropout이 OFF 된다.

model_scripted = torch.jit.script(model)  # TorchScript 형식으로 내보내기
model_scripted.save('./model/MODEL_PYTORCH_SCRIPT.pth')  # 저장하기

torch.save(model, './model/MODEL_PICKLE.pth')  # 저장하기

correct = 0
for data, target in test_loader:
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()
print('Test set Accuracy : {:.2f}%'.format(100. * correct / len(test_loader.dataset)))
