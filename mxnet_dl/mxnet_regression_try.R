require(mxnet)
rawdata<- read.csv("./data/processed6factor1dayRegression.csv")
#rawdata[,2]<- as.numeric(rawdata[,2])-1
rawdata[,c(3,4,5,6,7,8,9,10,11,12)]<- scale(rawdata[,c(3,4,5,6,7,8,9,10,11,12)],center = TRUE, scale = TRUE)
data <- mx.symbol.Variable("data")
train.x <- data.matrix(rawdata[c(1:200),c(3,4,5,6,7,8,9,10,11,12)])
train.y <- rawdata[c(1:200),2]
test.x <- data.matrix(rawdata[c(201:270),c(3,4,5,6,7,8,9,10,11,12)])
test.y <- rawdata[c(201:270),2]

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")
mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y,
                                     ctx=devices, num.round=10, array.batch.size=100,
                                     learning.rate=0.07, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))