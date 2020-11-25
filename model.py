import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
class Net(nn.Module):
 
  # our input image size (3, 640 , 480) 
  def __init__(self):
    super(Net, self).__init__()
    
    self.conv_1 = nn.Conv2d(3, 64, 7, stride=2, padding=1)
    self.conv_2 = nn.Conv2d(64, 128, 5, stride=2, padding=1)
    self.conv_2_bn = nn.BatchNorm2d(128)
    self.conv_3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
    self.conv_3_bn = nn.BatchNorm2d(256)
    self.conv_4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
    self.conv_4_bn = nn.BatchNorm2d(256)
    self.conv_5 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
    self.conv_5_bn = nn.BatchNorm2d(256)
    self.conv_6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
    self.conv_6_bn = nn.BatchNorm2d(256)
    self.conv_7 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
    # self.drop = nn.Dropout2d()
    self.fc1 = nn.Linear(20 * 15 * 256, 32)
    self.fc2 = nn.Linear(32, 4)
    
    
  def forward(self, x):
    
    x = F.relu(self.conv_1(x))
   
    x = F.relu(self.conv_2(x))
    x = self.conv_2_bn(x)
    x = F.relu(self.conv_3(x))
    x = self.conv_3_bn(x)
    x = F.relu(self.conv_4(x))
    x = self.conv_4_bn(x)
    x = F.relu(self.conv_5(x))
    x = self.conv_5_bn(x)
    x = F.relu(self.conv_6(x))
    x = self.conv_6_bn(x)
    x = F.relu(self.conv_7(x))
    # import pdb;pdb.set_trace()
    x = x.view(-1, 20 * 15 * 256)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    
    return x
  

if __name__ == '__main__':

  a = Variable(torch.ones((2, 3, 640, 480)))
  e = Variable(torch.ones((3, 3, 640, 480)))
  # print(a)
  model = Net()

  model(a)
  # import pdb;pdb.set_trace()
  print(model)