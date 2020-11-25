from model import Net
from dataset import grid_z
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np 
from functools import partial
from torch.autograd import Variable
# from tensorboardX import SummaryWriter
import torch
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = Net()
# model.to(device)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

checkpoint = torch.load('model_best.pth.tar')
model.load_state_dict(checkpoint["state_dict"])

train_data = grid_z(task='Test')
# val_data = grid_z(task='Val')
# length = len(data)s

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=10, shuffle=True, num_workers=4)
# val_loader = torch.utils.data.DataLoader(
# val_data, batch_size=32, shuffle=False, num_workers=4)



best_loss = float('inf')

tid = 0
are = np.empty((4,))
atexit.register(savewriter)
j =0 
model.eval()
with tqdm(total=len(val_loader)) as t:
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = torch.autograd.Variable(inputs, requires_grad=False)
        labels = torch.autograd.Variable(labels, requires_grad=False)
        labels = labels.type(torch.FloatTensor)
        # inputs, labels = inputs.to(device), labels.to(device)
        import pdb;pdb.set_trace()
        outputs = model(inputs)
        arr = outputs.clone().data.cpu().numpy()
        are = np.append(arr, are)

    t.set_description('Epoch ', str(epoch))

    t.set_postfix(loss=loss_val)

    t.update()