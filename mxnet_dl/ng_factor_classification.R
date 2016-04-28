require(mlbench)
require(mxnet)


rawdata<- read.csv("./ng_58factors_classification.csv")
#rawdata[,2]<- as.numeric(rawdata[,2])-1
rawdata[,c(3:ncol(rawdata))]<- scale(rawdata[,c(3:ncol(rawdata))],center = TRUE, scale = TRUE)

train.x <- data.matrix(rawdata[c(220:420), c(3,4,5,6,7,8,12,13,14,15,16,17,18,19,22,23,25,26,27,28)])
train.y <- rawdata[c(220:420),2]
test.x <- data.matrix(rawdata[c(421:480),  c(3,4,5,6,7,8,12,13,14,15,16,17,18,19,22,23,25,26,27,28)])
test.y <- rawdata[c(421:480),2]

mx.set.seed(0)
model <- mx.mlp(train.x, train.y, hidden_node=256, out_node=2, out_activation="softmax",
                num.round=10000, array.batch.size=50, learning.rate=0.01, momentum=0.9, 
                eval.metric=mx.metric.accuracy)


preds <- predict(model, test.x)
pred.label <- max.col(t(preds)) - 1
tt<-table(pred.label, test.y)

print((tt[1,1]+tt[2,2])/(tt[1,1]+tt[1,2]+tt[2,1]+tt[2,2]))

write.csv(data.frame(pred.label,test.y),"preds_mxnet.csv")
write.csv(data.frame(test.y),"testy_mxnet.csv")

plot(c(1:length(as.vector(pred.label))), as.vector(pred.label),col="green",xaxt='n',type ="l",ann=FALSE)
axis(1, at=label_inteval, labels=label_date,las=2)
lines(c(1:length(as.vector(test.y))),as.vector(test.y),type='l',col="red")
