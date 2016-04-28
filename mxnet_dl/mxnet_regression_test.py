import find_mxnet
import mxnet as mx
import pandas as pd
from random import random
import numpy as np
from math import pow
from sklearn import preprocessing

ng_data = pd.read_csv('./data/processed6factor1dayRegression.csv', sep=',', header = 1)
ng_data_array = ng_data.values

data_train, data_test  =  ng_data_array[:200,], ng_data_array[201:260,]
x_train = preprocessing.normalize(data_train[:,2:12],axis=0)
x_test = preprocessing.normalize(data_test[:,2:12],axis=0)
y_train = data_train[:,1]
y_test = data_test[:,1]


data = mx.symbol.Variable("data")
fc1 = mx.symbol.FullyConnected(data, num_hidden=64)
dp1 = mx.symbol.Dropout(fc1,name="dp1", p=0.1)
act1 = mx.symbol.Activation(dp1, name="relu1", act_type="relu")
fc2 = mx.symbol.FullyConnected(act1, num_hidden=32)
dp2 = mx.symbol.Dropout(fc2,name="dp2", p=0.1)
act2 = mx.symbol.Activation(dp2, name="relu2", act_type="relu")
fc3  = mx.symbol.FullyConnected(act2, num_hidden=1)
act3 = mx.symbol.Activation(fc3, name="relu3", act_type="relu")
lro = mx.symbol.LinearRegressionOutput(act3)
mx.set.seed(0)
model = mx.model.FeedForward.create(symbol = lro,ctx = mx.cpu(), num_epoch=10000, epoch_size=100,learning_rate=0.005, momentum=0.9, optimizer ="adam")

model.fit(X=x_train,eval_data=y_train,eval_metric=mx.metric.RMSE)
