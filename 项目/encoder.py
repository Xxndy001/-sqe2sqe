"""
进行编码
"""

import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
import config


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.ns),
                                     embedding_dim=50,
                                     padding_idx=config.ns.PAD
                                     )
        self.gru = nn.GRU(input_size=50,
                          hidden_size=64,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=False,
                          dropout=0)


    def forward(self, input,input_len):
        input_embeded = self.embedding(input)

        #对输入进行打包
        input_packed = pack_padded_sequence(input_embeded,input_len,batch_first=True)
        #经过GRU处理
        output,hidden = self.gru(input_packed)
        # print("encoder gru hidden:",hidden.size())
        #进行解包
        output_paded,seq_len = pad_packed_sequence(output,batch_first=True,padding_value=config.ns.PAD)
        return output_paded,hidden  #[1,batch_size,encoder_hidden_size]
