"""
进行模型的评估
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



def eval():
    model = Seq2Seq().to(config.device)
    model.load_state_dict(torch.load("./models/model.pkl"))

    loss_list = []
    acc_list = []
    data_loader = get_dataloader(train=False) #获取测试集
    with torch.no_grad():
        for idx,(input,target,input_len,target_len) in enumerate(data_loader):
            input = input.to(config.device)
            # target = target #[batch_size,max_len]
            input_len = input_len.to(config.device)
            #decoder_predict:[batch_size,max_len]
            decoder_outputs,decoder_predict = model.evaluate(input,input_len) #[batch_Size,max_len,vocab_size]
            loss = F.nll_loss(decoder_outputs.view(-1,len(config.ns)),target.to(config.device).view(-1),ignore_index=config.ns.PAD)
            loss_list.append(loss.item())

            #把traget 和 decoder_predict进行inverse_transform
            target_inverse_tranformed = [config.ns.inverse_transform(i) for i in target.numpy()]
            predict_inverse_tranformed = [config.ns.inverse_transform(i)for i in decoder_predict]
            cur_eq =[1 if target_inverse_tranformed[i] == predict_inverse_tranformed[i] else 0 for i in range(len(target_inverse_tranformed))]
            acc_list.extend(cur_eq)
            # print(np.mean(cur_eq))


    print("mean acc:{} mean loss:{:.6f}".format(np.mean(acc_list),np.mean(loss_list)))



def interface(_input): #进行预测
    model = Seq2Seq().to(config.device)
    model.load_state_dict(torch.load("./models/model.pkl"))
    input = list(str(_input))
    input_len = torch.LongTensor([len(input)]) #[1]
    input = torch.LongTensor([config.ns.transform(input)])  #[1,max_len]

    with torch.no_grad():
        input = input.to(config.device)
        input_len = input_len.to(config.device)
        _, decoder_predict = model.evaluate(input, input_len)  # [batch_Size,max_len,vocab_size]
        # decoder_predict进行inverse_transform
        pred = [config.ns.inverse_transform(i) for i in decoder_predict]
        print(_input,"---->",pred[0])


if __name__ == '__main__':
    interface("89767678")


