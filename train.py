from model import Net
from dataset import grid_z
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np 
from functools import partial
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = Net()
# model.to(device)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

train_data = grid_z(task='Train')
# val_data = grid_z(task='Val')
# length = len(data)s

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=256, shuffle=True, num_workers=4)
# val_loader = torch.utils.data.DataLoader(
# val_data, batch_size=32, shuffle=False, num_workers=4)

criterion_z = partial(nn.functional.mse_loss, size_average=False)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.001) 
scheduler = ReduceLROnPlateau(optimizer, 'min', cooldown=10)
optimize = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.005)

writer = SummaryWriter()



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def savewriter():
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()



best_loss = float('inf')

tid = 0
atexit.register(savewriter)
j =0 
model.train()
for epoch in range(200):
    with tqdm(total=len(train_loader)) as t:
        t.set_description('Epoch ', str(epoch))
        running_loss = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = torch.autograd.Variable(inputs, requires_grad=False)
            labels = torch.autograd.Variable(labels, requires_grad=False)
            labels = labels.type(torch.FloatTensor)
            # inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_outputs = outputs[:128,:,:,:]
            val_labels = labels[:128,:,:,:]
            train_outputs = outputs[128:,:,:,:]
            train_labels = labels[128:,:,:,:]
            # import pdb;pdb.set_trace()
            loss = criterion_z(train_outputs, train_labels)/(inputs.shape[0])
            import pdb;pdb.set_trace()  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss
            writer.add_scalar('data/train_loss', loss, tid)

            t.set_postfix(loss=loss)
            tid += 1
            t.update()

    running_loss = running_loss/len(train_loader)
    writer.add_scalar('data/train_running_loss', running_loss, tid)

    with tqdm(total=len(val_loader)) as t:
        t.set_description('Epoch ', str(epoch))

        eval_running_loss = 0
        model.eval()

        loss_val = criterion_z(val_outputs, val_labels)/(inputs.shape[0])

        eval_running_loss += loss_val

        t.set_postfix(loss=loss_val)

        t.update()

    eval_running_loss = eval_running_loss/len(val_loader)
    writer.add_scalar('data/eval_loss', eval_running_loss, tid)
    scheduler.step(eval_running_loss)



#save checkpoint 

    is_best = False
    if eval_running_loss < best_loss:
        print(eval_running_loss)
        is_best = True
        best_loss = eval_running_loss
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'train_loss': running_loss,
        'eval_loss': eval_running_loss,
        'optimizer': optimize.state_dict(),
    }
    save_checkpoint(checkpoint, is_best)
    del checkpoint