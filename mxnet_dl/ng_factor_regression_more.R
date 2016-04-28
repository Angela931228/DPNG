require(mxnet)


rawdata<- read.csv("./ng_58factors.csv")
#rawdata[,2]<- as.numeric(rawdata[,2])-1
rawdata[,c(3:ncol(rawdata))]<- scale(rawdata[,c(3:ncol(rawdata))],center = TRUE, scale = TRUE)

train.x <- data.matrix(rawdata[c(200:450), c(3,4,5,6,7,8,9,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29)])
train.y <- rawdata[c(200:450),2]
test.x <- data.matrix(rawdata[c(451:480), c(3,4,5,6,7,8,9,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29)])
test.y <- rawdata[c(451:480),2]

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, num_hidden=64)
dp1<- mx.symbol.Dropout(fc1,name="dp1", p=0.1)
act1 <- mx.symbol.Activation(dp1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, num_hidden=32)
dp2<- mx.symbol.Dropout(fc2,name="dp2", p=0.1)
act2 <- mx.symbol.Activation(dp2, name="relu2", act_type="relu")

fc3 <- mx.symbol.FullyConnected(act2, num_hidden=1)
act3 <- mx.symbol.Activation(fc3, name="relu3", act_type="relu")
lro <- mx.symbol.LinearRegressionOutput(act3)
mx.set.seed(0)

model <- mx.model.FeedForward.create(lro, train.x, train.y,ctx = mx.cpu(), num.round=15000, array.batch.size=100,
                                     learning.rate=0.005, momentum=0.9, eval.metric=mx.metric.rmse)
preds = predict(model, test.x)
sqrt(mean((preds-test.y)^2))

result_mse = sqrt(mean(((log(preds)-log(test.y))^2)))
print(result_mse)
sqrt(mean((diff(log(as.vector(test.y)))^2)))

t1<-sqrt(((log(preds)-log(test.y))^2))
t2<-sqrt((diff(log(as.vector(test.y)))^2))
write.csv(data.frame(as.vector(t1),c(as.vector(t2),1)),"rmse.csv")
label_inteval<-seq(from=1,to=length(as.vector(preds)), by=3)
label_date<-rawdata[c(451:480),1][label_inteval]
plot(c(1:length(as.vector(preds))), as.vector(preds),col="green",xaxt='n',type ="l", ylim=c(1.5,2.4),ann=FALSE)
axis(1, at=label_inteval, labels=label_date,las=2)
lines(c(1:length(as.vector(test.y))),as.vector(test.y),type='l',col="red")
write.csv(data.frame(rawdata[c(451:490),1],as.vector(preds),as.vector(test.y)),"./data/pred_test_close_2.csv")
plot(as.vector(preds)-as.vector(test.y),col="red",xaxt='n',type ="p",ann=FALSE)
axis(1, at=label_inteval, labels=label_date,las=2)
abline(a=0,b=0,col="green")

