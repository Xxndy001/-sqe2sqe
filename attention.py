import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import pickle
import tensorflow as tf
import math
# 导入原始训练数据
def get_datas(file = "datas\\translate.csv",nums = None):
    all_datas = pd.read_csv(file)
    en_datas = list(all_datas["english"])
    ch_datas = list(all_datas["chinese"])

    if nums == None:
        return en_datas,ch_datas
    else:
        return en_datas[:nums],ch_datas[:nums]


class MyDataset(Dataset):
    def __init__(self,en_data,ch_data,en_word_2_index,ch_word_2_index):
        self.en_data = en_data
        self.ch_data = ch_data
        self.en_word_2_index = en_word_2_index
        self.ch_word_2_index = ch_word_2_index

    def __getitem__(self,index):
        en = self.en_data[index]
        ch = self.ch_data[index]

        en_index = [self.en_word_2_index[i] for i in en]
        ch_index = [self.ch_word_2_index[i] for i in ch]

        return en_index,ch_index


    def batch_data_process(self,batch_datas):
        global device
        en_index , ch_index = [],[]
        en_len , ch_len = [],[]

        for en,ch in batch_datas:
            en_index.append(en)
            ch_index.append(ch)
            en_len.append(len(en))
            ch_len.append(len(ch))

        max_en_len = max(en_len)
        max_ch_len = max(ch_len)

        en_index = [ i + [self.en_word_2_index["<PAD>"]] * (max_en_len - len(i))   for i in en_index]
        ch_index = [[self.ch_word_2_index["<BOS>"]]+ i + [self.ch_word_2_index["<EOS>"]] + [self.ch_word_2_index["<PAD>"]] * (max_ch_len - len(i))   for i in ch_index]

        en_index = torch.tensor(en_index,device = device)
        ch_index = torch.tensor(ch_index,device = device)


        return en_index,ch_index


    def __len__(self):
        assert len(self.en_data) == len(self.ch_data)
        return len(self.ch_data)

# # attention
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, Q, K, V):
        K_transpose = torch.transpose(K, 1, 2)
        # K_transpose.shape = (batch_size, K_size, seq_len)
        d = K_transpose.size(1)
        QK = torch.bmm(Q, K_transpose) / math.sqrt(d)
        # QK.shape = QK_softmax=(batch_size, m, seq_len)
        QK_softmax = torch.softmax(QK, dim=2)
        # V.shape = (batch_size, seq_len,V_size)

        # result.shape = (batch_size, m, V_size)
        result = torch.bmm(QK_softmax, V)
        return result



import torch
import torch.nn as nn

# class SelfAttention(nn.Module):
#     def __init__(self, hidden_size, num_heads, dropout):
#         super(SelfAttention, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.head_size = hidden_size // num_heads
#         self.dropout = nn.Dropout(dropout)
#
#         self.qkv_projection = nn.Linear(hidden_size, hidden_size * 3)
#         self.output_projection = nn.Linear(hidden_size, hidden_size)
#
#     def forward(self, x, mask=None):
#         batch_size, seq_len, hidden_size = x.size()
#         qkv = self.qkv_projection(x).view(batch_size, seq_len, 3, self.num_heads, self.head_size).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
#         if mask is not None:
#             attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
#         attn_probs = nn.Softmax(dim=-1)(attn_scores)
#         attn_probs = self.dropout(attn_probs)
#
#         attn_output = torch.matmul(attn_probs, v)
#         attn_output = attn_output.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_len, -1)
#         attn_output = self.output_projection(attn_output)
#
#         return attn_output



class Encoder(nn.Module):
    def __init__(self,encoder_embedding_num,encoder_hidden_num,en_corpus_len):
        super().__init__()
        self.embedding = nn.Embedding(en_corpus_len,encoder_embedding_num)
        self.lstm = nn.LSTM(encoder_embedding_num,encoder_hidden_num,batch_first=True)

    def forward(self,en_index):
        en_embedding = self.embedding(en_index)
        _,encoder_hidden =self.lstm(en_embedding)

        return encoder_hidden



class Decoder(nn.Module):
    def __init__(self,decoder_embedding_num,decoder_hidden_num,ch_corpus_len):
        super().__init__()
        self.embedding = nn.Embedding(ch_corpus_len,decoder_embedding_num)
        self.lstm = nn.LSTM(decoder_embedding_num,decoder_hidden_num,batch_first=True)

    def forward(self,decoder_input,hidden):
        embedding = self.embedding(decoder_input)
        decoder_output,decoder_hidden = self.lstm(embedding,hidden)

        return decoder_output,decoder_hidden


def translate(sentence):
    global en_word_2_index,model,device,ch_word_2_index,ch_index_2_word
    en_index = torch.tensor([[en_word_2_index[i] for i in sentence]],device=device)

    result = []
    encoder_hidden = model.encoder(en_index)
    decoder_input = torch.tensor([[ch_word_2_index["<BOS>"]]],device=device)

    decoder_hidden = encoder_hidden
    while True:
        decoder_output,decoder_hidden = model.decoder(decoder_input,decoder_hidden)
        pre = model.classifier(decoder_output)

        w_index = int(torch.argmax(pre,dim=-1))
        word = ch_index_2_word[w_index]

        if word == "<EOS>" or len(result) > 50:
            break

        result.append(word)
        decoder_input = torch.tensor([[w_index]],device=device)

    print("译文: ","".join(result))



# 预测
class Seq2Seq(nn.Module):
    def __init__(self,encoder_embedding_num,encoder_hidden_num,en_corpus_len,decoder_embedding_num,decoder_hidden_num,ch_corpus_len):
        super().__init__()
        self.encoder = Encoder(encoder_embedding_num,encoder_hidden_num,en_corpus_len)
        self.decoder = Decoder(decoder_embedding_num,decoder_hidden_num,ch_corpus_len)
        self.classifier = nn.Linear(decoder_hidden_num,ch_corpus_len)

        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self,en_index,ch_index):
        decoder_input = ch_index[:,:-1]
        label = ch_index[:,1:]

        encoder_hidden = self.encoder(en_index)
        decoder_output,_ = self.decoder(decoder_input,encoder_hidden)

        pre = self.classifier(decoder_output)
        loss = self.cross_loss(pre.reshape(-1,pre.shape[-1]),label.reshape(-1))

        return loss



if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
# 词典导入
#  word_2_index,index_2_word的类型分别是字典和列表，后者是列表因为其默认序号可以替代数字，因而省略了数字部分。
    #  提取出的四个序列如下所示，此处并不是构造one-hot向量，而是作为数字索引，在之后直接构造对应的embbding。
    #  两个pickle文件中各自有三列数据，第一列是字向量，因此只需要提取后两列数据即可。

    with open("datas\\ch.vec","rb") as f1:
        _, ch_word_2_index,ch_index_2_word = pickle.load(f1)

    with open("datas\\en.vec","rb") as f2:
        _, en_word_2_index, en_index_2_word = pickle.load(f2)

    ch_corpus_len = len(ch_word_2_index)
    en_corpus_len = len(en_word_2_index)

    ch_word_2_index.update({"<PAD>":ch_corpus_len,"<BOS>":ch_corpus_len + 1 , "<EOS>":ch_corpus_len+2})
    en_word_2_index.update({"<PAD>":en_corpus_len})

    ch_index_2_word += ["<PAD>","<BOS>","<EOS>"]
    en_index_2_word += ["<PAD>"]

    ch_corpus_len += 3
    en_corpus_len = len(en_word_2_index)


    en_datas,ch_datas = get_datas(nums=500)
    encoder_embedding_num = 50
    encoder_hidden_num = 100
    decoder_embedding_num = 107
    decoder_hidden_num = 100

    batch_size = 2
    epoch = 40
    lr = 0.001

    dataset = MyDataset(en_datas,ch_datas,en_word_2_index,ch_word_2_index)
    dataloader = DataLoader(dataset,batch_size,shuffle=False,collate_fn = dataset.batch_data_process)

    model = Seq2Seq(encoder_embedding_num,encoder_hidden_num,en_corpus_len,decoder_embedding_num,decoder_hidden_num,ch_corpus_len)
    model = model.to(device)


    opt = torch.optim.Adam(model.parameters(),lr = lr)

    for e in range(epoch):
        for en_index,ch_index  in dataloader:
            loss = model(en_index,ch_index)
            loss.backward()
            opt.step()
            opt.zero_grad()

        print(f"loss:{loss:.3f}")


    while True:
        s = input("请输入英文: ")
        translate(s)