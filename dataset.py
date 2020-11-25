from torchvision import transforms
from torch.utils.data.dataset import Dataset
import os
import numpy as np
from PIL import Image
import json
from matplotlib import pyplot as plt
import torch
import numpy as np
import pickle as pkl
normalize = transforms.Normalize(mean=[0.6695, 0.6534, 0.6378],
                                 std=[0.0509, 0.055, 0.061])
# transformations = transforms.ToTensor()
transformations = transforms.Compose([transforms.ToTensor(), normalize])
# transform_image_show = transforms.Resize([256, 256])

folder = "./gdrive/My Drive/flipkart/"
imagefolder = "./image/"

class grid_z(Dataset):
    def __init__(
            self,
            transformation=transformations,
            task='Train'
    ):

        data_z = {}
        if task == 'Train':
            with open(folder + 'train.pkl', 'rb') as f:
                raw_data = pkl.load(f)
        elif task == 'Test':
            with open(folder + 'test.pkl', 'rb') as f:
                raw_data = pkl.load(f)
        else:
            with open('val.pkl', 'rb') as f:
                raw_data = pkl.load(f)
                # print('val', len(raw_data))

        # with open(os.path.join(img_path, '/home-local2/shared/deepskymodel/sun360z_exp4.json'), 'r') as f:
        #     z = json.load(f)

        self.keys = list(raw_data.keys())

        self.data = raw_data

        self.transformation = transformation
        self.task = task

    def __len__(self):
        length = len(self.keys)
        return length

    def __getitem__(self, index):
        label = np.array(self.data[self.keys[index]])
        img = self.keys[index]
        keys = self.keys[index]

        img = Image.open(os.path.join(imagefolder + '/images', img))
        # import pdb;pdb.set_trace()
        if self.transformation is not None:
            img = self.transformation(img)
        return (img, label, keys)



if __name__ == '__main__':
    if True:
        from imageio import imread
        custom_dataset = grid_z(task="Test")


        length = len(custom_dataset)
        print(custom_dataset[0])
        # import pdb;pdb.set_trace()
        # length1 = len(im_gen)
        # print(length, length1)
    # print(len(z_data))
    # print(len(z))
    # for i, data in enumerate(z):
    #     print('z_vector', data[1])
    #     print('azimuth', data[2])
    #     import pdb
    #     pdb.set_trace()
