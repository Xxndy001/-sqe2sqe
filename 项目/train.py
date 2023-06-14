"""
进行模型的训练
"""
import torch
import torch.nn.functional as F
from seq2seq import Seq2Seq
from torch.optim import Adam
from dataset import get_dataloader
from tqdm import tqdm
import config
import numpy as np
import pickle
from matplotlib import pyplot as plt
from eval import eval

model = Seq2Seq().to(config.device)

optimizer = Adam(model.parameters())

loss_list = []

def train(epoch):
    data_loader = get_dataloader(train=True)
    bar = tqdm(data_loader,total=len(data_loader))

    for idx,(input,target,input_len,target_len) in enumerate(bar):
        input = input.to(config.device)
        target = target.to(config.device)
        input_len = input_len.to(config.device)
        optimizer.zero_grad()
        decoder_outputs = model(input,input_len) #[batch_Size,max_len,vocab_size]
        loss = F.nll_loss(decoder_outputs.view(-1,len(config.ns)),target.view(-1),ignore_index=config.ns.PAD)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        bar.set_description("epoch:{} idx:{} loss:{:.6f}".format(epoch,idx,np.mean(loss_list)))

        if idx%100 == 0:
            torch.save(model.state_dict(),"./models/model.pkl")
            torch.save(optimizer.state_dict(),"./models/optimizer.pkl")
            pickle.dump(loss_list,open("./models/loss_list.pkl","wb"))


if __name__ == '__main__':
    for i in range(5):
        train(i)
        eval()

    plt.figure(figsize=(50,8))
    plt.plot(range(len(loss_list)),loss_list)
    plt.show()

