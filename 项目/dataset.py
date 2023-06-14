"""
准备数据集
"""
from torch.utils.data import DataLoader,Dataset
import numpy as np
import config
import torch

class NumDataset(Dataset):
    def __init__(self,train=True):
        np.random.seed(9) if train else np.random.seed(10)
        self.size = 400000 if train else 100000
        self.data = np.random.randint(1,1e8,size=self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input = list(str(self.data[idx]))
        target = input+["0"]
        return input,target,len(input),len(target)

def collate_fn(batch):
    """
    :param batch:[(一个getitem的结果)，(一个getitem的结果)，(一个getitem的结果)、、、、]
    :return:
    """
    #把batch中的数据按照input的长度降序排序
    batch = sorted(batch,key=lambda x:x[-2],reverse=True)
    input,target,input_len,target_len = zip(*batch)
    input = torch.LongTensor([config.ns.transform(i,max_len=config.max_len) for i in input])
    target = torch.LongTensor([config.ns.transform(i,max_len=config.max_len,add_eos=True) for i in target])
    input_len = torch.LongTensor(input_len)
    target_len = torch.LongTensor(target_len)
    return input,target,input_len,target_len

def get_dataloader(train=True):
    batch_size = config.train_batchsize if train else config.test_batch_size
    return DataLoader(NumDataset(train),batch_size=batch_size,shuffle=False,collate_fn=collate_fn)


if __name__ == '__main__':
    loader = get_dataloader(train=False)
    for idx,(input,target,input_len,target_len) in enumerate(loader):
        print(idx)
        print(input)
        print(target)
        print(input_len)
        print(target_len)
        break
