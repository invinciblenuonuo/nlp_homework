import os
import numpy as np
import torch 
import torch.nn as nn

from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

def ReadData(train_or_test, num=None):
    # 打开文件，文件路径通过 os.path.join 构建，使用 utf-8 编码方式打开
    with open(os.path.join( train_or_test + ".txt"), encoding="utf-8") as f:
        # 读取文件内容并按行分割，得到一个包含所有行的列表all_data
        all_data = f.read().split("\n")

    #初始化两个空列表，用于存储文本和标签
    texts  = []
    labels = []

    # 遍历每一行数据
    for data in all_data:
        # 检查行是否非空
        if data: 
            # 使用制表符('\t')分割行数据为文本和标签
            t, l = data.split("\t")
            # 将文本和标签添加到各自的列表中
            texts.append(t)
            labels.append(l)

    # 如果没有传递 num 参数，则返回所有的文本和标签
    if num is None:
        return texts, labels
    # 如果传递了 num 参数，则返回前 num 个文本和标签
    else:
        return texts[:num], labels[:num]
    
#构建字典
def BuildCorpus(train_texts, embedding_num):
    # 初始化一个字典，用于将单词映射到索引。
    word_2_index = {"<PAD>": 0, "<UNK>": 1}
    # 遍历训练文本中的每个文本
    for text in train_texts:
        # 遍历文本中的每个单词
        for word in text:
            # 如果单词不在字典中，将其映射到当前字典的长度（即当前最大索引+1）
            word_2_index[word] = word_2_index.get(word, len(word_2_index))
    # 返回构建好的字典和一个pytorch的嵌入层，该层的输入大小为字典的长度，输出大小为指定的嵌入数目
    return word_2_index, nn.Embedding(len(word_2_index), embedding_num)

#继承自Dataset类，是pytorch里面的数据集类
class TextDataset(Dataset):
    def __init__(self,all_text,all_label,word_2_index,max_len):
        self.all_text=all_text
        self.all_label=all_label
        self.word_2_index=word_2_index
        self.max_len=max_len

    def __getitem__(self, index):
        text=self.all_text[index][:self.max_len] 
        #截断在max_len,并获取索引为index的文本数据
        label=int(self.all_label[index])
        #获取索引为index的标签数据
        text_idx=[self.word_2_index.get(i,1) for i in text]
        #列表推导式，把test中的每一个字都转成index，并检验text中的字符在不在字典里，不在用1代替
        text_idx=text_idx+[0]*(self.max_len-len(text_idx))
        #填充，保证长度一致
        text_idx=torch.tensor(text_idx).unsqueeze(dim=0)
        #Tensor 对象可以利用 GPU 进行加速计算,为了加快计算，使用了tensor转化
        return text_idx,label

    def __len__(self):
        return len(self.all_text)
    

class Block(nn.Module):
    def __init__(self,kernel_s,embeddin_num,max_len,hidden_num):
        super().__init__()
        self.cnn=nn.Conv2d(in_channels=1,out_channels=hidden_num,kernel_size=(kernel_s,embeddin_num))#创建卷积层
                # batch * in_channel * len * emb_num
                # 1     *   1      *    7    *    5
        self.act=nn.ReLU()#激活函数ReLU

        self.mxp=nn.MaxPool1d(kernel_size=(max_len-kernel_s+1))#池化
    
    def forward(self,batch_emb):
        c_out=self.cnn.forward(batch_emb)
        a_out=self.act.forward(c_out)
        a_out=a_out.squeeze(dim=-1)
        m_out=self.mxp.forward(a_out)
        m_out=m_out.squeeze(dim=-1)
        return m_out

#最重要的模型构建方法：
class TextCNNModel(nn.Module):
    def __init__(self,emb_matrix,max_len,hidden_num):
        super().__init__()

        self.emb_num=emb_matrix.weight.shape[1] #获取矩阵的列数，即字维度

        self.block1=Block(2,self.emb_num,max_len,hidden_num)
        self.block2=Block(3,self.emb_num,max_len,hidden_num)
        self.block3=Block(4,self.emb_num,max_len,hidden_num) 
        self.block4=Block(5,self.emb_num,max_len,hidden_num) 
        self.block5=Block(6,self.emb_num,max_len,hidden_num) 

        #三个块的目的是去捕捉不同长度文本的特征

        self.emb_matrix=emb_matrix
        self.classifier=nn.Linear(hidden_num*5,class_num)
        self.loss_fun=nn.CrossEntropyLoss()#交叉熵损失函数，包含softmax

    def forward(self,batch_idx,batch_label=None):
        batch_emb=self.emb_matrix(batch_idx) #根据index转成matrix
        b1_result=self.block1.forward(batch_emb)
        b2_result=self.block2.forward(batch_emb)
        b3_result=self.block3.forward(batch_emb)
        b4_result=self.block4.forward(batch_emb)
        b5_result=self.block5.forward(batch_emb)

        feature=torch.cat([b1_result,b2_result,b3_result,b4_result,b5_result],dim=1)
        pre    =self.classifier(feature)
        
        if batch_label is not None:
            loss=self.loss_fun(pre,batch_label)
            return loss
        else:
            return torch.argmax(pre,dim=-1)




if __name__ == "__main__":
    print("start")
    
    #读入训练集
    train_text,train_label=ReadData("train",100) 
    dev_text,dev_label=ReadData("test",100) 
    
    embedding=5     #字向量维度
    max_len=20       #句子最大长度
    batch_size=1    #批次大小
    epoch=100       #循环次数
    lr=0.001        #学习率
    hidden_num=10    #隐藏层个数

    class_num=len(set(train_label))
    #使用GPU
    device="cuda:0" if torch.cuda.is_available() else "cpu" 
    print(device)
    #构建字典
    word_2_index,word_embedding=BuildCorpus(train_text,embedding)
    #加载数据集
    train_dataset=TextDataset(train_text,train_label,word_2_index,max_len)
    train_loader=DataLoader(train_dataset,batch_size,shuffle=False)
    #加载测试集
    dev_dataset=TextDataset(dev_text,dev_label,word_2_index,max_len)
    dev_loader=DataLoader(train_dataset,batch_size,shuffle=False)
    #模型入口
    model = TextCNNModel(word_embedding,max_len,hidden_num)
    #优化器定义
    #model.parameters()表示要优化的参数是模型中的所有可学习参数
    opt=torch.optim.AdamW(model.parameters(),lr=lr)

    mode=input("选择模式： 1 训练 2 测试")
    if mode=="1":
        print("train start")
        #循环训练的核心部分 
        for e in range(epoch):
            for batch_idx,batch_label in train_loader:
                batch_idx=batch_idx.to(device)
                batch_label=batch_label.to(device)
                loss=model.forward(batch_idx,batch_label)
                loss.backward()
                opt.step()
                opt.zero_grad()
        

            #准确率测试
            right_num=0
            for batch_idx,batch_label in dev_loader:
                batch_idx=batch_idx.to(device)
                batch_label=batch_label.to(device)
                pre=model.forward(batch_idx)
                right_num+=int(torch.sum(pre==batch_label))#求算对的个数
                print(pre)
            print(f"loss:{loss:.3f} acc = {(right_num/len(dev_text)*100):.2f}%")
            
        torch.save(model, 'model.pt')

    else:
        model=torch.load('model.pt')
        user_text,user_label=ReadData("user",1) 
        user_dataset=TextDataset(user_text,user_label,word_2_index,max_len)
        user_loader=DataLoader(user_dataset,1,shuffle=False)

        for batch_idx,batch_label in user_loader:
            batch_idx=batch_idx.to(device)
            batch_label=batch_label.to(device)
            pre=model.forward(batch_idx)
            print(pre)






    



