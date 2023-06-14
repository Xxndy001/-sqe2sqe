"""
数字序列化方法
"""

class NumSequence:
    """
    input : intintint
    output :[int,int,int]
    """
    PAD_TAG = "<PAD>"
    UNK_TAG = "<UNK>"
    SOS_TAG = "<SOS>"
    EOS_TAG = "<EOS>"

    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

    def __init__(self):
        self.dict = {
            self.PAD_TAG:self.PAD,
            self.UNK_TAG: self.UNK,
            self.SOS_TAG: self.SOS,
            self.EOS_TAG: self.EOS
        }
        #0--》int ,1--->int,2--->int
        for i in range(0,10):
            self.dict[str(i)] = len(self.dict)
        self.inverse_dict = dict(zip(self.dict.values(),self.dict.keys()))

    def transform(self,sentence,max_len=None,add_eos=False):
        """
        实现转化为数字序列
        :param sentence: list() ,["1","2","5"...str]
        :param max_len: int
        :param add_eos: 是否要添加结束符
        :return: [int,int,int]

        """

        if add_eos : #不是必须的，仅仅是为了最终句子的长度=设置的max；如果没有，最终的句子长度= max_len+1
            max_len = max_len - 1
        if max_len is not None:
            if len(sentence)> max_len:
                sentence = sentence[:max_len]
            else:
                sentence = sentence + [self.PAD_TAG]*(max_len-len(sentence))
        if add_eos:
            if sentence[-1] == self.PAD_TAG:  #句子中有PAD，在PAD之前添加EOS
                pad_index = sentence.index(self.PAD_TAG)
                sentence.insert(pad_index,self.EOS_TAG)
            else:#句子中没有PAD，在最后添加EOS
                sentence += [self.EOS_TAG]

        return [self.dict.get(i,self.UNK) for i in sentence]

    def inverse_transform(self,incides):
        """
        把序列转化为数字
        :param incides:[1,3,4,5,2,]
        :return: "12312312"
        """
        result = []
        for i in incides:
            temp = self.inverse_dict.get(i, self.UNK_TAG)
            if temp != self.EOS_TAG:  #把EOS之后的内容删除，123---》1230EOS，predict 1230EOS123
                result.append(temp)
            else:
                break

        return "".join(result)

    def __len__(self):
        return len(self.dict)


if __name__ == '__main__':
    num_Sequence = NumSequence()
    print(num_Sequence.dict)
    s = list("123123")
    ret = num_Sequence.transform(s)
    print(ret)
    ret = num_Sequence.inverse_transform(ret)
    print(ret)