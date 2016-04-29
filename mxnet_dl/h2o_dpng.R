library("zoo")
library("tseries")
library("ggplot2")
library("EMD")
library("xlsx")
library("e1071")
library("hydroGOF")

setwd("C:/Users/angela.zhou/Desktop/angela_dissertation/market_state/")
library(h2o)
h2o.init(nthreads = -1)
h2o.removeAll()
ng_data<- h2o.importFile("./ng_58factors.csv")
ng_data_csv <- read.csv("./ng_58factors.csv")
ng_6yeardata_csv <- read.csv("ng_6year_data.csv")



plot(c(1:length(ng_data_csv[,1])),ng_data_csv[,2],type='l', xaxt='n',ann=FALSE)
label_inteval<-seq(from=1,to=length(ng_data_csv[,1]), by=9)
label_date<-ng_data_csv[,1][label_inteval]
axis(1, at=label_inteval, labels=label_date,las=2)
abline(a=0,b=0,col="green")

hyper_params <- list(
  hidden=list(c(64,32),c(32),c(100),c(200),c(200,100)),
  input_dropout_ratio=c(0,0.01,0.05),
  rate=c(0.01,0.02,0.001,0.005),
  rate_annealing=c(1e-8,1e-7,1e-6)
)

mms<-{}
for(i in 1:120){
  ng_train <- ng_data[c((i+1):(320+i)),]
  ng_valid <- ng_data[c((i+320):(360+i)),]
  ng_test <- ng_data[c(361+i),]
  ng_hex<- ng_data
  best_mse<-99
  model_grid <- h2o.grid("deeplearning",x=colnames(ng_hex)[c(3,4,5,6,7,8,9,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29)],y=colnames(ng_hex)[2],hyper_params = hyper_params, training_frame = ng_train, validation_frame = ng_test, export_weights_and_biases=T)
  for(model_id in model_grid@model_ids){
  model <- h2o.getModel(model_id)
  mse <- h2o.mse(model,valid =TRUE)
    if(best_mse> mse){
      best_model<-model
      best_mse<-mse
    }
    print(model_id)
    print(sprintf("Test set MSE: %f", mse))
  }
  mm<-h2o.predict(best_model, ng_hex[c(361+i),c(3,4,5,6,7,8,9,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29)])
  mms<- append(mms,mm)
}

plot(c(1:length(as.vector(mm))), as.vector(ng_hex[c(461:490),2]),type='l',col="red", ylim=c(1.6,2.4))
lines(c(1:length(as.vector(mm))),as.vector(mm),type='l',col="green")


model_ng<- h2o.deeplearning(x=colnames(ng_hex)[c(3,4,5,6,7,8,9,13,14,15,16,17,18,19,20,23,24,26,27,28,29)],l1=1e-7 ,rate=0.02, input_dropout_ratio=0.05,y=colnames(ng_hex)[2], hidden = c(200), training_frame = ng_train, validation_frame = ng_test)
summary(model_ng)
print(model_ng)
mm<-h2o.predict(model_ng, ng_hex[c(261:293),c(3,4,5,6,7,8,9,13,14,15,16,17,18,19,20,23,24,26,27,28,29)])

result_mse = sqrt(mean(((log(mm)-log(ng_hex[c(261:293),c(2)]))^2)))
print(result_mse)
sqrt(mean((diff(log(as.vector(ng_hex[c(261:293),c(2)])))^2)))

rmse(as.vector(mm), as.vector(ng_hex[c(261:293),c(2)]))
plot(c(1:length(as.vector(mm))), as.vector(ng_hex[c(261:293),2]),type='l',col="green")
lines(c(1:length(as.vector(mm))),as.vector(mm),type='l',col="red")
plot(ng_data_csv[,1],ng_data_csv[,2],type='l')
