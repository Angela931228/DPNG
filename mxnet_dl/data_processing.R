library(XLConnect)
wb<- loadWorkbook("./data/ng_factors.xlsx")
lst = readWorksheet(wb, sheet = getSheets(wb))
t<- lapply(lst, function(x) x[,c(1:2)])

c_names<- names(lst)

ng1_price <-  data.frame(t[1])
colnames(ng1_price) <- c("date","price")
#ng1_price$date<- as.Date(ng1_price$date,format="%m/%d%Y")
factors<-{}
for(i in 3:nrow(ng1_price)){
  currentDate<- ng1_price$date[i]
  factor<-{}
  for(k in 1:length(t)){
    ticker<- data.frame(t[k])
    colnames(ticker) <- c("date","price")
    #ticker$date<- as.Date(ticker$date,format="%Y-%m-%d")
    indexes<- which(ticker$date < currentDate)
    factor<- cbind(factor,ticker$price[indexes[length(indexes)]])
  } 
  factors<- rbind(factors,factor)
}

result<-cbind(ng1_price[c(-1,-2),],factors)

names(result)<- c("date","price","ytd_price",c_names[-1])

write.csv(result,"ng_1day_prediction_1.csv")
