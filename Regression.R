library(simex)
library(caret)
library(tidyverse)
library(sandwich)
library(lmtest)
library(quantreg)
library(ggplot2)
library(AER)

data_full <- read.csv("../../Data/Data_prepared.csv",stringsAsFactors = FALSE)
data_full$AdjLocation<-factor(data_full$AdjLocation)

#Get test set
data_test <- data_full %>% 
  filter(train == FALSE , Image_present == 1) %>% 
  mutate(l_size = log(Size),l_price = log(Price)) %>% 
  droplevels()
data_test[,c("l_size","No_bathrooms","No_bedrooms","No_Images","VGG_Sentiment","Desc_length","ML_text_sentiment")] <- scale(data_test[,c("l_size","No_bathrooms","No_bedrooms","No_Images","VGG_Sentiment","Desc_length","ML_text_sentiment")])
f_ctrl <- formula("l_price~l_size+No_bathrooms+No_bedrooms+ AdjLocation")

#Compare model Control and M1
coef_l <- list()
AIC_l <- BIC_l <- r.sq_l<-list()

additions <- c("","VGG_Sentiment + No_Images")

#Baseline model + Image present
for(k in 1:length(additions)){
  if(k == 1){
    g <- f_ctrl
  }else{
    g <- update.formula(f_ctrl,paste0(".~.+",additions[k]))
  }
  m <- lm(g, data = data_test)
  AIC_l[[k]] <- AIC(m)
  BIC_l[[k]] <- BIC(m)
  r.sq_l[[k]] <- summary(m)$adj.r.squared
  u<-unclass(coeftest(m,vcov. = sandwich))
  colnames(u)[2]<-"se"
  temp <- as.data.frame(u[-grep(pattern = "Location",row.names(u)),c(1,2)])
  if(k == 1){
    temp <- cbind(temp, Parameter = row.names(temp), Model = rep("Control",nrow(temp)))
  }
  else{
    temp <- cbind(temp, Parameter = row.names(temp), Model = rep("M 1",nrow(temp)))
  }
  row.names(temp) <- NULL
  coef_l[[k]] <- temp
  rm(u,temp,m)
}


#Plot
coef <- do.call(rbind,coef_l)%>% filter(!Parameter %in% c("(Intercept)")) %>% mutate (count = 1)
coef$Parameter <- as.factor(ifelse(coef$Parameter == "VGG_Sentiment",
                                   "Image sentiment", 
                                   ifelse(coef$Parameter == "No_Images", "Number of images", 
                                          ifelse(coef$Parameter == "l_size","Size",
                                                 ifelse(coef$Parameter == "No_bathrooms","Number of bathrooms","Number of bedrooms")))))
ggplot(coef,aes(x= fct_reorder(Parameter,.x = count,sum),Estimate)) + 
  geom_hline(yintercept = 0, lty = 2, lwd = 0.5,color="grey") + 
  geom_errorbar(aes(ymin = Estimate - 1.96*se , 
                    ymax = Estimate + 1.96*se , 
                    color = Model),
                lwd=1, 
                width=0,
                position = position_dodge(width = 0.5)) + 
  geom_point(aes(color = Model),
             position = position_dodge(width = 0.5),
             shape = 21,fill="white",
             size = 2.5) +
  scale_color_brewer(type="qual",palette = 6)+
  xlab("Variable")+
  ylab("Standardized coefficient") + 
  coord_flip()+ 
  theme_bw() + 
  theme(legend.position="top", 
        legend.title = element_blank(),
        axis.text=element_text(size=18),
        legend.text = element_text(size=18),
        axis.title=element_text(size=18,face="bold"))

#Comparison Image sentiment vs. textual descriptions, i.e., M1 and M2

coef_l <- list()
AIC_l <- BIC_l <- r.sq_l <-list()

additions <- c("VGG_Sentiment + No_Images","VGG_Sentiment + ML_text_sentiment +  Desc_length + No_Images")
#Baseline model + Image present
for(k in 1:length(additions)){
  g <- update.formula(f_ctrl,paste0(".~.+",additions[k]))
  m <- lm(g, data = data_test)
  AIC_l[[k]] <- AIC(m)
  BIC_l[[k]] <- BIC(m)
  r.sq_l[[k]] <- summary(m)$adj.r.squared
  u<-unclass(coeftest(m,vcov. = sandwich))
  colnames(u)[2]<-"se"
  temp <- as.data.frame(u[-grep(pattern = "Location",row.names(u)),c(1,2)])
  temp <- cbind(temp, Parameter = row.names(temp), Model = rep(paste0("M ",k),nrow(temp)))
  row.names(temp) <- NULL
  coef_l[[k]] <- temp
  rm(u,temp)
}

#Plot

coef <- do.call(rbind,coef_l) %>% filter(!Parameter %in% c("(Intercept)","l_size","No_bedrooms","No_bathrooms")) %>% mutate (count = 1)
coef$Parameter <- ifelse(coef$Parameter == "VGG_Sentiment",
                         "Image sentiment", ifelse(coef$Parameter == "Desc_length","Description length",
                                                   ifelse(coef$Parameter == "ML_text_sentiment","Description sentiment","Number of images")))


ggplot(coef,aes(fct_reorder(Parameter,.x = count,sum),Estimate)) + 
  geom_hline(yintercept = 0, lty = 2, lwd = 0.5,color="grey") + 
  geom_errorbar(aes(ymin = Estimate - 1.96*se , 
                    ymax = Estimate + 1.96*se , 
                    color = Model),
                lwd=1, 
                width=0,
                position = position_dodge(width = 0.5)) + 
  geom_point(aes(color = Model),
             position = position_dodge(width = 0.5),
             shape = 21,fill="white",
             size = 2.5) +
  scale_color_brewer(type="qual", palette = 6)+
  ylab("Standardized coefficient") + 
  xlab("Variable") +
  coord_flip()+ 
  theme_bw() + 
  theme(legend.position="top", 
        legend.title = element_blank(),
        legend.text = element_text(size = 18),
        axis.text=element_text(size=18),
        axis.title=element_text(size=18,face="bold"))




#Prediction of log price
#---------------------------------------------------------------------------------
set.seed(08052019)
inTraining <- createDataPartition(data_test$l_price, p = .75, list = FALSE)
fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)

rmse_l <- list()
rmse_l[[1]] <- sd(data_test$l_price[-inTraining])
for(k in 1:3){
  f <- switch(k, f_ctrl, 
              formula("l_price ~ VGG_Sentiment"),
              update.formula(f_ctrl,".~. + VGG_Sentiment")) 
  
  X <- model.matrix(f,data_test)
  y <- data_test$l_price
  lmFIT <- train(y = y[inTraining], x = X[inTraining,],
                 method = "lm", 
                 trControl = fitControl,
                 tuneLength = 2)
  
  rmse_l[[length(rmse_l) + 1]] <- RMSE(y[-inTraining],predict(lmFIT,X[-inTraining,]))
  print(paste("Iteration",k,":","Starting SVM"))
  svmFIT <- train(y = y[inTraining], x = X[inTraining,],
                  method = "svmRadial", 
                  trControl = fitControl,
                  tuneLength = 10)
  
  rmse_l[[length(rmse_l) + 1]] <- RMSE(y[-inTraining],predict(svmFIT,X[-inTraining,]))
  print(paste("Iteration",k,":","Starting RF"))
  rfFIT <- train(y = y[inTraining], x = X[inTraining,],
                 method = "rf", 
                 trControl = fitControl,
                 tuneLength = 8)
  rmse_l[[length(rmse_l) + 1]] <- RMSE(y[-inTraining],predict(rfFIT,X[-inTraining,]))
}

#IV Estimation
#---------------------------------------------------------------------------------
#IV estimation with instrument (a)
m_IV <- ivreg(l_price ~ l_size + No_bedrooms + No_bathrooms + AdjLocation + VGG_Sentiment |. -VGG_Sentiment + Average_Blue ,data = data_test)

summary(m_IV, diagnostics = TRUE)
#IV estimation with instrument (b)

m_IV <- ivreg(l_price ~ l_size + No_bedrooms + No_bathrooms + AdjLocation + VGG_Sentiment |. -VGG_Sentiment + Lagged_VGG ,data = data_test)

summary(m_IV, diagnostics = TRUE)


#IV estimation with both instruments
m_IV <- ivreg(l_price ~ l_size + No_bedrooms + No_bathrooms + AdjLocation + VGG_Sentiment |. -VGG_Sentiment + Lagged_VGG + Average_Blue ,data = data_test)
summary(m_IV, diagnostics = TRUE)

#Robustness Checks
#---------------------------------------------------------------------------------

#Quantile Regression
#---------------------------------------------------------------------------------
f <- formula("Price ~ Size + No_bedrooms + No_bathrooms + AdjLocation + VGG_Sentiment")
quantiles <- c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
qr_l <- list()
X <- model.matrix(f,data_test)
for(k in 1:length(quantiles)){
  print(k)
  m <- rq(f,tau=quantiles[k], data = data_test, method = "br",model=TRUE)
  QR.b <- boot.rq(x = X, 
                  y = data_test$Price,tau=quantiles[k],bsmethod = "wxy",R = 500)
  qr_l[[k]] <- c(m$coefficients["VGG_Sentiment"],quantile(QR.b$B[,ncol(QR.b$B)],c(0.025,0.975)))
}
qr_coef <- do.call(rbind,qr_l)
qr_coef <- cbind(qr_coef,quantiles)
colnames(qr_coef) <- c("Estimate","Lower","Upper","Quantile")
qr_coef <- as.data.frame(qr_coef)

#Plotting the graphic
ggplot(qr_coef,aes(x = Quantile)) + 
  geom_ribbon(aes(ymin =Lower,ymax = Upper),fill="gray90") + 
  geom_line(aes(y = Estimate),color="black") + 
  ylab("Standardized coefficients") +
  scale_x_continuous(breaks = seq(0.1, 0.9, by = 0.1)) + 
  theme_bw() + 
  theme(axis.text=element_text(size=18),
        axis.title=element_text(size=18,face="bold"))
#Non-linear relationship
#---------------------------------------------------------------------------------
AIC_l <- BIC_l <- r.sq_l <-list()

additions <- c("VGG_Sentiment",
               "I(VGG_Sentiment^2)",
               "I(VGG_Sentiment^3)",
               "VGG_Sentiment + I(VGG_Sentiment^2)", 
               "VGG_Sentiment + I(VGG_Sentiment^3)",
               "VGG_Sentiment + I(VGG_Sentiment^2) + I(VGG_Sentiment^3)")
#Baseline model + Image present
for(k in 1:length(additions)){
  g <- update.formula(f_ctrl,paste0(".~.+",additions[k]))
  m <- lm(g, data = data_test)
  AIC_l[[k]] <- AIC(m)
  BIC_l[[k]] <- BIC(m)
  r.sq_l[[k]] <- summary(m)$adj.r.squared
  rm(g,m)
}
#SIMEX correction
#---------------------------------------------------------------------------------
vgg_err<-RMSE(data_test$VGG_Sentiment,data_test$Residual)
f <- update.formula(f_ctrl,". ~ . + VGG_Sentiment")
m_naive <- lm(f,data = data_test, x = TRUE, y = TRUE)
m_sim<-simex(m_naive,
             "VGG_Sentiment", 
             asymptotic = FALSE, 
             measurement.error = vgg_err,
             lambda = c(0.5,1,1.5,2,2.5,3))
res<-m_sim$residuals
N<-length(res)
loglike <- 0.5*(-N*(log(2*pi)+1-log(N)+log(sum(res^2))))
aic_sim <- -2*loglike+2*m_naive$rank
bic_sim <- -2*loglike+log(N)*m_naive$rank
r_sq <- 1-sum(res^2)/((N-1)*var(data_test$l_price))
adj_r_sq<-1-(1-r_sq)*(N-1)/(N-m_naive$rank)


