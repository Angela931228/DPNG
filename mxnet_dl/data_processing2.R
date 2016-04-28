ng_factors <-read.csv("ng_1day_prediction_1.csv")
tftcs <-read.csv("./data/tftcdata.csv")
ng_factors$date<- as.Date(ng_factors$date,format="%m/%d/%Y")
tftcs$date<- as.Date(tftcs$date,format="%m/%d/%Y")


factors<-{}
for(i in 1:nrow(ng_factors)){
  factor<-{}
  currentDate<- ng_factors$date[i]
  indexes<- which(tftcs$date < currentDate)
  factor<- cbind(ng_factors[i,],tftcs[indexes[length(indexes)],-1])
  factors<- rbind(factors,factor)
}

write.csv(factors,"ng_58factors.csv")
