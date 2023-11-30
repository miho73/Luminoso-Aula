import matplotlib.pyplot as plt
import numpy as np
import torch

import cv2

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

print('Current cuda device is', device)

model = torch.jit.load('/Users/yeji/dev/Luminoso-Project/Luminoso-Aula/ml/model/Model_PYTORCH_SCRIPT_NL.pth')
model.eval()


def predict_ml(image, w, h):
    image = np.array(image).astype(np.float32).reshape(w, h)
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    data = torch.from_numpy(np.array(image)).float()
    data = data.reshape(1, 1, 28, 28)
    data = data.to(device)
    output = model(data)
    prediction = output.data.max(1)[1]
    res = output.data.detach().cpu().numpy()
    return res[0].tolist(), prediction.item()
