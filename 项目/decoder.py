"""
实现解码器
"""
import torch.nn as nn
import config
import torch
import torch.nn.functional as F
import numpy as np


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()

        self.embedding = nn.Embedding(num_embeddings=len(config.ns),
                                      embedding_dim=50,
                                      padding_idx=config.ns.PAD)

        #需要的hidden_state形状：[1,batch_size,64]
        self.gru = nn.GRU(input_size=50,
                          hidden_size=64,
                          num_layers=1,
                          bidirectional=False,
                          batch_first=True,
                          dropout=0)

        #假如encoder的hidden_size=64，num_layer=1 encoder_hidden :[2,batch_sizee,64]

        self.fc = nn.Linear(64,len(config.ns))

    def forward(self, encoder_hidden):

        #第一个时间步的输入的hidden_state
        decoder_hidden = encoder_hidden  #[1,batch_size,encoder_hidden_size]
        #第一个时间步的输入的input
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.LongTensor([[config.ns.SOS]]*batch_size).to(config.device)         #[batch_size,1]
        # print("decoder_input:",decoder_input.size())


        #使用全为0的数组保存数据，[batch_size,max_len,vocab_size]
        decoder_outputs = torch.zeros([batch_size,config.max_len,len(config.ns)]).to(config.device)

        for t in range(config.max_len):
            decoder_output_t,decoder_hidden = self.forward_step(decoder_input,decoder_hidden)
            decoder_outputs[:,t,:] = decoder_output_t


            #获取当前时间步的预测值
            value,index = decoder_output_t.max(dim=-1)
            decoder_input = index.unsqueeze(-1)  #[batch_size,1]
            # print("decoder_input:",decoder_input.size())
        return decoder_outputs,decoder_hidden


    def forward_step(self,decoder_input,decoder_hidden):
        '''
        计算一个时间步的结果
        :param decoder_input: [batch_size,1]
        :param decoder_hidden: [batch_size,encoder_hidden_size]
        :return:
        '''

        decoder_input_embeded = self.embedding(decoder_input)
        # print("decoder_input_embeded:",decoder_input_embeded.size())

        out,decoder_hidden = self.gru(decoder_input_embeded,decoder_hidden)

        #out ：【batch_size,1,hidden_size】

        out_squeezed = out.squeeze(dim=1) #去掉为1的维度
        out_fc = F.log_softmax(self.fc(out_squeezed),dim=-1) #[bathc_size,vocab_size]
        # out_fc.unsqueeze_(dim=1) #[bathc_size,1,vocab_size]
        # print("out_fc:",out_fc.size())
        return out_fc,decoder_hidden

    def evaluate(self,encoder_hidden):

        # 第一个时间步的输入的hidden_state
        decoder_hidden = encoder_hidden  # [1,batch_size,encoder_hidden_size]
        # 第一个时间步的输入的input
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.LongTensor([[config.ns.SOS]] * batch_size).to(config.device)  # [batch_size,1]
        # print("decoder_input:",decoder_input.size())

        # 使用全为0的数组保存数据，[batch_size,max_len,vocab_size]
        decoder_outputs = torch.zeros([batch_size, config.max_len, len(config.ns)]).to(config.device)

        decoder_predict = []  #[[],[],[]]  #123456  ,targe:123456EOS,predict:123456EOS123
        for t in range(config.max_len):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[:, t, :] = decoder_output_t

            # 获取当前时间步的预测值
            value, index = decoder_output_t.max(dim=-1)
            decoder_input = index.unsqueeze(-1)  # [batch_size,1]
            # print("decoder_input:",decoder_input.size())
            decoder_predict.append(index.cpu().detach().numpy())

        #返回预测值
        decoder_predict = np.array(decoder_predict).transpose() #[batch_size,max_len]
        return decoder_outputs, decoder_predict

