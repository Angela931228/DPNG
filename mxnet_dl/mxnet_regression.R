require(mlbench)
require(mxnet)


data(Sonar, package = "mlbench")

rawdata<- read.csv("./data/processed6factor1dayRegression.csv")
#rawdata[,2]<- as.numeric(rawdata[,2])-1
rawdata[,c(3,4,5,6,7,8,9,10,11,12)]<- scale(rawdata[,c(3,4,5,6,7,8,9,10,11,12)],center = TRUE, scale = TRUE)

train.x <- data.matrix(rawdata[c(1:200),c(3,4,5,6,7,8,9,10,11,12)])
train.y <- rawdata[c(1:200),2]
test.x <- data.matrix(rawdata[c(201:260),c(3,4,5,6,7,8,9,10,11,12)])
test.y <- rawdata[c(201:260),2]

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, num_hidden=50)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc1 <- mx.symbol.FullyConnected(data, num_hidden=1)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
lro <- mx.symbol.LinearRegressionOutput(act1)
mx.set.seed(0)
model <- mx.model.FeedForward.create(lro, train.x, train.y,ctx=mx.cpu(), num.round=3000, array.batch.size=10,
                                     learning.rate=2e-4, momentum=0.9, eval.metric=mx.metric.rmse)
preds = predict(model, test.x)
sqrt(mean((preds-test.y)^2))

result_mse = sqrt(mean(((log(preds)-log(test.y))^2)))
sqrt(mean((diff(log(as.vector(test.y)))^2)))
print(result_mse)
label_inteval<-seq(from=1,to=length(as.vector(preds)), by=4)
label_date<-rawdata[c(201:260),1][label_inteval]
plot(c(1:length(as.vector(preds))), as.vector(preds),col="green",xaxt='n',type ="l",ann=FALSE)
axis(1, at=label_inteval, labels=label_date,las=2)
lines(c(1:length(as.vector(test.y))),as.vector(test.y),type='l',col="red")
