import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import scipy.io as io
import torchvision
from torchvision import transforms as T
import numpy as np
def MyLoader(path,type):
    if type=='img':
        return Image.open(path).convert('RGB')
    elif type=='vector':
        return np.load(path, allow_pickle=True)
    elif type=='msi':
        return io.loadmat(path)['msi']




class Mydataset(Dataset):
    def __init__(self, txt, transform_rs=None, transform_sv=None, loader=MyLoader):
        with open(txt,'r',encoding="utf-8") as fh:
            file=[]
            for line in fh:
                line=line.strip('\n')
                line=line.rstrip()
                words=line.split("\t")

                file.append((words[0], words[1], words[2], words[3], words[4], int(words[-1]))) # 路径1 路径2 路径3 路径4 路径5 标签


        self.file=file
        self.transform_rs = transform_rs
        self.transform_sv = transform_sv
        self.loader=loader


    def __getitem__(self, index):

        hrs, sv0, sv1, sv2, sv3, label = self.file[index]

        hrs_f = self.loader(hrs,type='img')
        sv_f0 = self.loader(sv0, type='img')
        sv_f1 = self.loader(sv1, type='img')
        sv_f2 = self.loader(sv2, type='img')
        sv_f3 = self.loader(sv3, type='img')


        if self.transform_rs is not None:
            hrs_f = self.transform_rs(hrs_f)
        if self.transform_sv is not None:
            sv_f0 = self.transform_sv(sv_f0)
            sv_f1 = self.transform_sv(sv_f1)
            sv_f2 = self.transform_sv(sv_f2)
            sv_f3 = self.transform_sv(sv_f3)


        return hrs_f, sv_f0, sv_f1, sv_f2, sv_f3, label

    def __len__(self):
        return len(self.file)


if __name__ == "__main__":
    pass