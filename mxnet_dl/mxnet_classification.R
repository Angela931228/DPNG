require(mlbench)
require(mxnet)


data(Sonar, package = "mlbench")

rawdata<- read.csv("./data/processed6factor1day.csv")
#rawdata[,2]<- as.numeric(rawdata[,2])-1
rawdata[,c(3,4,5,6,7,8,9,10,11,12)]<- scale(rawdata[,c(3,4,5,6,7,8,9,10,11,12)],center = TRUE, scale = TRUE)
train.x <- data.matrix(rawdata[c(1:200),c(3,4,5,6,7,8,9,10,11,12)])
train.y <- rawdata[c(1:200),2]
test.x <- data.matrix(rawdata[c(201:260),c(3,4,5,6,7,8,9,10,11,12)])
test.y <- rawdata[c(201:260),2]
mx.set.seed(0)
model <- mx.mlp(train.x, train.y, hidden_node=100, out_node=2, out_activation="softmax",
                num.round=5000, array.batch.size=15, learning.rate=0.005, momentum=0.9, 
                eval.metric=mx.metric.accuracy)
graph.viz(model$symbol$as.json())

preds <- predict(model, test.x)
pred.label <- max.col(t(preds)) - 1
tt<-table(pred.label, test.y)

print((tt[1,1]+tt[2,2])/(tt[1,1]+tt[1,2]+tt[2,1]+tt[2,2]))

write.csv(data.frame(pred.label,test.y),"preds_mxnet.csv")
write.csv(data.frame(test.y),"testy_mxnet.csv")

plot(c(1:length(as.vector(pred.label))), as.vector(pred.label),col="green",xaxt='n',type ="l",ann=FALSE)
axis(1, at=label_inteval, labels=label_date,las=2)
lines(c(1:length(as.vector(test.y))),as.vector(test.y),type='l',col="red")
