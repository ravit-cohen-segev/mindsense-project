setwd("C:/Users/Amit Saad/Dropbox/MATRIX/Potential")
rm(list = ls())
graphics.off() # close all figures, if any exist
library(wmwpow)
library(WMWssp)
wmwpowd(n=9, m=9, distn="norm(0.4,0.14)",distm="norm(0.2, 0.14)", sides="greater",alph=0.05, nsims=10000)
wmwpowd(n=9, m=9, distn="beta(2,5)",distm="beta(2,5)", sides="greater",alph=0.05, nsims=10000)

WMWssp(0.4, 0.2, alpha = 0.05, power = 0.8, t = 1/2,
       simulation = FALSE, nsim = 10^4)


#data <- read.csv ("matrix_uri.csv")
data <- read.csv ("C://Users/Amit Saad/Dropbox/MATRIX/Potential/potential_data_040719.csv",stringsAsFactors = FALSE)
impor<- read.csv ("C://Users/Amit Saad/Dropbox/MATRIX/Potential/varibale importance.csv",stringsAsFactors = FALSE)
data = data[1:18,]

impor=table(impor)
good=subset(data,data$Ouctome=="Good")
poor=subset(data,data$Ouctome=="Poor")

#Con/Inco ratio- Total#
#Potential#
wilcox.test(data$con_incon_ratio_ALL_echoes_P~data$Ouctome,alternative = "greater")
median(good$con_incon_ratio_ALL_echoes_P)
median(poor$con_incon_ratio_ALL_echoes_P)


wilcox.test(data$con_incon_ratio_T_echo_P_P~data$Ouctome,alternative = "greater")
median(good$con_incon_ratio_T_echo_P_P)
median(poor$con_incon_ratio_T_echo_P_P)

wilcox.test(data$con_incon_ratio_P_echo_T_P~data$Ouctome,alternative = "greater")
median(good$con_incon_ratio_P_echo_T_P)
median(poor$con_incon_ratio_P_echo_T_P)

#Content#
wilcox.test(data$con_incon_ratio_ALL_echoes_C~data$Ouctome,alternative = "greater")
median(good$con_incon_ratio_ALL_echoes_C)
median(poor$con_incon_ratio_ALL_echoes_C)

x=(poor$con_incon_ratio_ALL_echoes_C)
y=x+median(x)*0.34
WMWssp(x, y, alpha = 0.05, t = 1/2,p=0.8,
       simulation =FALSE, nsim = 10000)


mean(good$con_incon_ratio_ALL_echoes_C)
sd(good$con_incon_ratio_ALL_echoes_C)
mean(poor$con_incon_ratio_ALL_echoes_C)
sd(poor$con_incon_ratio_ALL_echoes_C)

wmwpowd(n=9, m=9, distn="norm(0.3999,0.06867)",distm="norm(0.3959237,0.09525749)", sides="greater",alph=0.05, nsims=10000)


wilcox.test(data$con_incon_ratio_T_echo_P_C~data$Ouctome,alternative = "greater")
median(good$con_incon_ratio_T_echo_P_C)
median(poor$con_incon_ratio_T_echo_P_C)

wilcox.test(data$con_incon_ratio_P_echo_T_C~data$Ouctome,alternative = "greater")
median(good$con_incon_ratio_P_echo_T_C)
median(poor$con_incon_ratio_P_echo_T_C)

#Inter-relation#
wilcox.test(data$con_incon_ratio_ALL_echoes_I~data$Ouctome,alternative = "greater")
median(good$con_incon_ratio_ALL_echoes_I)
median(poor$con_incon_ratio_ALL_echoes_I)

mean(good$con_incon_ratio_ALL_echoes_I)
sd(good$con_incon_ratio_ALL_echoes_I)
mean(poor$con_incon_ratio_ALL_echoes_I)
sd(poor$con_incon_ratio_ALL_echoes_I)

wmwpowd(n=9, m=9, distn="norm(0.508,0.1421586)",distm="norm(0.4149434, 0.1141422)", sides="greater",alph=0.05, nsims=10000)



wilcox.test(data$con_incon_ratio_T_echo_P_I~data$Ouctome,alternative = "greater")
median(good$con_incon_ratio_T_echo_P_I)
median(poor$con_incon_ratio_T_echo_P_I)

wilcox.test(data$con_incon_ratio_P_echo_T_I~data$Ouctome,alternative = "greater")
median(good$con_incon_ratio_P_echo_T_I)
median(poor$con_incon_ratio_P_echo_T_I)


#boxplot- sum of the con/incon of the potential
pdf("Following the potential.pdf", width=7.9, height=7.9)
#potential#
boxplot(data$con_incon_ratio_ALL_echoes_P~data$Ouctome,data=data, col=c("red","blue"),main="The congruence/incongruence ratio of the potential- Total",xlab="treatment outcome",ylab="The con/incong ratio of the potential per treatment")
legend (2,25, c("poor treatments","good treatment"),fill=c("red","blue"))
boxplot(data$con_incon_ratio_P_echo_T_P~data$Ouctome,data=data,col=c("red","blue"),main="The congruence/incongruence ratio of the potential-patient",xlab="treatment outcome",ylab="The con/incong ratio of the potential per treatment")
legend (2,10, c("poor treatments","good treatment"),fill=c("red","blue"))
boxplot(data$con_incon_ratio_T_echo_P_P~data$Ouctome,data=data,col=c("red","blue"),main="The congruence/incongruence ratio of the potential-therapist",xlab="treatment outcome",ylab="The con/incong ratio of the potential per treatment")
legend (2,15, c("poor treatments","good treatment"),fill=c("red","blue"))

#content#
boxplot(data$con_incon_ratio_ALL_echoes_C~data$Ouctome,data=data, col=c("red","blue"),main="The congruence/incongruence ratio of the content- Total",xlab="treatment outcome",ylab="The con/incong ratio of the content per treatment")
legend (2,25, c("poor treatments","good treatment"),fill=c("red","blue"))
boxplot(data$con_incon_ratio_P_echo_T_C~data$Ouctome,data=data,col=c("red","blue"),main="The congruence/incongruence ratio of the content-patient",xlab="treatment outcome",ylab="The con/incong ratio of the content per treatment")
legend (2,10, c("poor treatments","good treatment"),fill=c("red","blue"))
boxplot(data$con_incon_ratio_T_echo_P_C~data$Ouctome,data=data,col=c("red","blue"),main="The congruence/incongruence ratio of the content-therapist",xlab="treatment outcome",ylab="The con/incong ratio of the content per treatment")
legend (2,15, c("poor treatments","good treatment"),fill=c("red","blue"))

#Inter-relation#
boxplot(data$con_incon_ratio_ALL_echoes_I~data$Ouctome,data=data, col=c("red","blue"),main="The congruence/incongruence ratio of the inter-relation- Total",xlab="treatment outcome",ylab="The con/incong ratio of the inter-relation per treatment")
legend (2,25, c("poor treatments","good treatment"),fill=c("red","blue"))
boxplot(data$con_incon_ratio_P_echo_T_I~data$Ouctome,data=data,col=c("red","blue"),main="The congruence/incongruence ratio of the inter-relation-patient",xlab="treatment outcome",ylab="The con/incong ratio of the inter-relation per treatment")
legend (2,10, c("poor treatments","good treatment"),fill=c("red","blue"))
boxplot(data$con_incon_ratio_T_echo_P_I~data$Ouctome,data=data,col=c("red","blue"),main="The congruence/incongruence ratio of the inter-relation-therapist",xlab="treatment outcome",ylab="The con/incong ratio of the inter-relation per treatment")
legend (2,15, c("poor treatments","good treatment"),fill=c("red","blue"))

dev.off()

png(file = "Following the potential_1.jpeg", width = 300, height = 210,units= "mm", res=300)
par(mar=c(7,5,4,2),mfrow=c(1,3))
boxplot(data$con_incon_ratio_ALL_echoes_P~data$Ouctome,data=data, cex.main=2.5, cex.lab=2.5,cex.axis=1.5, col=c("red","blue"),main="Potential",xlab="treatment outcome",ylab="The con/incong ratio of the potential per treatment")
legend (0.5,0.46, cex=1.5, c("poor treatments","good treatment"),fill=c("blue","red"))
boxplot(data$con_incon_ratio_ALL_echoes_C~data$Ouctome,data=data, cex.main=2.5, cex.lab=2.5,cex.axis=1.5, col=c("red","blue"),main="Content",xlab="treatment outcome")
legend (2,25, c("poor treatments","good treatment"),fill=c("red","blue"))
boxplot(data$con_incon_ratio_ALL_echoes_I~data$Ouctome,data=data, cex.main=2.5, cex.lab=2.5,cex.axis=1.5, col=c("red","blue"),main="Interrelation",xlab="treatment outcome")
legend (2,25, c("poor treatments","good treatment"),fill=c("red","blue"))
dev.off()

png(file = "Variable importance.jpeg", width = 300, height = 210,units= "mm", res=300)
par(mar=c(7,5,4,2),mfrow=c(1,3))
barplot (impor)

##############################
#GLM#
data$Ouctome[which(data$Ouctome=="Poor")] = 0
data$Ouctome[which(data$Ouctome=="Good")] = 1
data$Ouctome = as.numeric(data$Ouctome)

m1 = glm(data$Ouctome ~ data$con_incon_ratio_ALL_echoes_P+data$con_incon_ratio_ALL_echoes_I, family = binomial)
m0 = glm(data$Ouctome ~ 1, family = binomial)

100*(m1$null.deviance-m1$deviance)/m1$null.deviance
pchisq(deviance(m0)-deviance(m1),
       df.residual(m0)-df.residual(m1),
       lower.tail=FALSE)

summary(m1)
anova(m1,test="Chisq")


require(relaimpo)
formula = "data$Ouctome ~ data$con_incon_ratio_ALL_echoes_P + data$con_incon_ratio_ALL_echoes_C +
data$con_incon_ratio_ALL_echoes_I"
model = lm(formula, data = data)
relimp.res = calc.relimp(model)

#plot- relative importance#
plot_relaimpo = function(relimp.res, filename, labs, sort = TRUE){
  x = as.data.frame(relimp.res@lmg) 
  names(x) = "lmg"
  if (sort==TRUE){
    x$lmg = sort(x$lmg, decreasing = T) # sort in a descending order
  }
  r = round(relimp.res@R2,2) #R^2 value
  jpeg(file = "Following the potential.jpeg", width = 300, height = 210,units= "mm", res=300)
  par(mar=c(7,5,4,2))
  y = barplot(x$lmg, ylim = c(0,max(x$lmg)+0.05), ylab = "Variance explained", cex.lab = 1.5, cex.axis = 1.5)
  text(y,y=-0.02, labs, xpd=TRUE, cex = 1.8)
  tmp_lbs = round(x$lmg,digits=2)
  tmp_lbs[which(tmp_lbs==0)] = "<0.01"
  text(y,x$lmg+0.01,labels=tmp_lbs,cex=2)
  legend("top",legend = bquote(paste("Total ",R^2," = ",.(r))), box.col="white", cex=2)
  dev.off()
}
