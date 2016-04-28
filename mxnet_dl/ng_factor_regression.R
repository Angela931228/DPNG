require(mxnet)


rawdata<- read.csv("./ng_1day_prediction_1.csv")
#rawdata[,2]<- as.numeric(rawdata[,2])-1
rawdata[,c(3:ncol(rawdata))]<- scale(rawdata[,c(3:ncol(rawdata))],center = TRUE, scale = TRUE)

train.x <- data.matrix(rawdata[c(220:420), c(3,4,5,6,7,8,12,13,14,15,16,17,18,19,22,23,25,26,27)])
train.y <- rawdata[c(220:420),2]
test.x <- data.matrix(rawdata[c(421:480),  c(3,4,5,6,7,8,12,13,14,15,16,17,18,19,22,23,25,26,27)])
test.y <- rawdata[c(421:480),2]

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, num_hidden=64)
dp1<- mx.symbol.Dropout(fc1,name="dp1", p=0.2)
act1 <- mx.symbol.Activation(dp1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, num_hidden=32)
dp2<- mx.symbol.Dropout(fc2,name="dp2", p=0.2)
act2 <- mx.symbol.Activation(dp2, name="relu2", act_type="relu")

fc3 <- mx.symbol.FullyConnected(act2, num_hidden=1)
act3 <- mx.symbol.Activation(fc3, name="relu3", act_type="relu")
lro <- mx.symbol.LinearRegressionOutput(act3)
mx.set.seed(0)
model <- mx.model.FeedForward.create(lro, train.x, train.y,ctx = mx.cpu(), num.round=20000, array.batch.size=100,
                                     learning.rate=0.01, momentum=0.9, eval.metric=mx.metric.rmse)
preds = predict(model, test.x)
sqrt(mean((preds-test.y)^2))

result_mse = sqrt(mean(((log(preds)-log(test.y))^2)))
print(result_mse)
sqrt(mean((diff(log(as.vector(test.y)))^2)))

label_inteval<-seq(from=1,to=length(as.vector(preds)), by=4)
label_date<-rawdata[c(201:260),1][label_inteval]
plot(c(1:length(as.vector(preds))), as.vector(preds),col="green",xaxt='n',type ="l",, ylim=c(1.6,2.6),ann=FALSE)
axis(1, at=label_inteval, labels=label_date,las=2)
lines(c(1:length(as.vector(test.y))),as.vector(test.y),type='l',col="red")

