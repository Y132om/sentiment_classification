#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import pandas as pd
import numpy as np
import jieba

from keras.layers import Dense,Input,Flatten,Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM,GRU



'''
读入文本，进行分词，并转换为字符串
'''

#读入数据
neg=pd.read_excel('./data/neg.xls',header=None)
pos=pd.read_excel('./data/pos.xls',header=None)

#合并语料
pn=pd.concat([pos,neg],ignore_index=True)
#计算语料数目
neglen=len(neg)
poslen=len(pos)

#定义分词函数
cw=lambda x:list(jieba.cut(x))
pn['words']=pn[0].apply(cw)

#一行数据最多的词汇数
max_document_length=max([len(x) for x in pn['words']])
texts=[' '.join(x) for x in pn['words']]

'''
对读入的字符串，进行处理。并转换为编号
字典形式
'''
#实例化分词器，设置字典中最大词汇数30000
tokenizer=Tokenizer(num_words=30000)
tokenizer.fit_on_texts(texts)
sequences=tokenizer.texts_to_sequences(texts)
#把序列设定为10000的长度，超过10000的部分舍弃，不到10000则补0
sequences=pad_sequences(sequences,maxlen=1000)
sequences=np.array(sequences)
dict_text=tokenizer.word_index


'''
准备模型训练需要的数据
'''
#定义标签
positive_labels=[[0,1] for _ in range(poslen)]
negative_labels=[[1,0] for _ in range(neglen)]
y=np.concatenate([positive_labels,negative_labels],0) #横向拼接
#打乱数据
np.random.seed(10)
#permutation:对原来的数组进行重新洗牌,返回一个新的打乱顺序的数组，并不改变原来的数组
shuffle_indices=np.random.permutation((np.arange(len(y))))
x_shuffled=sequences[shuffle_indices]
y_shuffled=y[shuffle_indices]
#数据集切分成两部分
test_sample_index=-1*int(0.1*float(len(y)))
x_train,x_test=x_shuffled[:test_sample_index],x_shuffled[test_sample_index:]
y_train,y_test=y_shuffled[:test_sample_index],y_shuffled[test_sample_index:]

'''
网络模型结构
采用函数式模型
'''
#输入层
sequence_input=Input(shape=(1000,))
embedding_layer=Embedding(30000,128,input_length=1000)
embedding_sequences=embedding_layer(sequence_input)
#LSTM
lstm1=LSTM(10,return_sequences=True,dropout=0.2,recurrent_dropout=0.2)(embedding_sequences)
lstm1=Flatten()(lstm1)
lstm1=Dense(16,activation='relu')(lstm1)
lstm1=Dropout(0.5)(lstm1)

#双向LSTM
lstm2=Bidirectional(LSTM(10,return_sequences=True,dropout=0.2,recurrent_dropout=0.2))(embedding_sequences)
lstm2=Flatten()(lstm2)
lstm2=Dense(16,activation='relu')(lstm2)
lstm2=Dropout(0.5)(lstm2)

#GRU
lstm3=GRU(10,return_sequences=True,dropout=0.2,recurrent_dropout=0.2)(embedding_sequences)
lstm3=Flatten()(lstm3)
lstm3=Dense(16,activation='relu')(lstm3)
lstm3=Dropout(0.5)(lstm3)


#输出层
preds=Dense(2,activation='softmax')(lstm2)

'''
模型设置
'''
model=Model(sequence_input,preds)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=128,epochs=5,validation_data=(x_test,y_test))

'''
预测
'''
def predict(text):
    cw=list(jieba.cut(text))
    word_id=[]

    for word in cw:
        try:
            temp=dict_text[word]
            word_id.append(temp)
        except:
            word_id.append(0)

    word_id=np.array(word_id)
    #这一位置增加一个一维，这一位置指的是np.newaxis所在的位置
    word_id=word_id[np.newaxis,:]
    '''
    sequences：浮点数或整数构成的两层嵌套列表
    maxlen：None或整数，为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0.
    dtype：返回的numpy array的数据类型
    padding：‘pre’或‘post’，确定当需要补0时，在序列的起始还是结尾补`
    truncating：‘pre’或‘post’，确定当需要截断序列时，从起始还是结尾截断
    value：浮点数，此值将在填充时代替默认的填充值0
    '''
    sequences=pad_sequences(word_id,maxlen=1000,padding='post')
    result=np.argmax(model.predict(sequences))
    if(result==1):
        print('positive comment')
    else:
        print('negative comment')

predict("东西质量不错，下次还来购买")


