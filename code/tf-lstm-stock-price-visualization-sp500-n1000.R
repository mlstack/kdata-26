#!/usr/bin/env /c/Apps/Anaconda3/Scripts/Rscript

# RNN - Feature Label Map
# modified
# Author : Yibeck Lee(yibeck.lee@gmail.com)
devtools::install_github("rstudio/tensorflow")
library(tensorflow)
install_tensorflow()
library(ggplot2)

fileName = 'sp500-n1000.csv'
data = read.csv(fileName, header = F)[1:1000, 1]

# data = scale(data)
#tsData = rbind(
#     data.frame(seq = 1:length(data), data = data, cls = 'sp500'))
#ggplot(tsData, aes(seq, data)) + geom_line(aes(colour = cls))
# data = seq(1, 50, 1)

numObs = length(data)
trainObsRatio = 0.8
trainObsEnd = as.integer(numObs * trainObsRatio)
trainData = data[1:trainObsEnd]
testData = data[as.integer(trainObsEnd+1):as.integer(numObs)]

# RNN parameter
inputDim = as.integer(1L)
seqLength = as.integer(5L)
hiddenDim = as.integer(100L)
print(sprintf('dim %f', hiddenDim))

# trainData의 feature와 label 할당
train_x = c()
train_y = c()
for (i in 1:(length(trainData) - seqLength)) {
    # 1 2 3 4 5   2 3 4 5 6  3 4 5 6 7 
    train_x = c(train_x, trainData[i:(i + seqLength - 1)])
    # 2 3 4 5 6   3 4 5 6 7  4 5 6 7 8
    train_y = c(train_y, trainData[(i + 1):(i + seqLength)])
}
# print(train_x[1:50])

length(train_x)
length(train_y)
length(trainData)-seqLength
# train_x

array(train_x, dim = c(5, i, 1))
dim(array(train_x, dim = c(5, i, 1)))
aperm(array(data = train_x, dim = c(5, i, 1)), c(2, 1, 3))
train_x = aperm(array(data=train_x,dim=c(5, i, 1)), c(2, 1, 3))
# train_x
train_y = matrix(train_y, ncol = 5)

matrix(train_y, ncol = 5)
train_y = matrix(train_y, ncol = 5, byrow = T)
# train_y

# testData의 feature와 label 할당
test_x = c()
test_y = c()
for (i in 1:(length(testData) - seqLength)) {
    test_x = c(test_x, testData[i:(i + seqLength - 1)])
    test_y = c(test_y, testData[(i + 1):(i + seqLength)])
}

test_x = aperm(array(test_x, dim = c(5, i, 1)), c(2, 1, 3))
# test_x

test_y = matrix(test_y, ncol = 5, byrow = T)
# test_y
print(data.frame(train_x))
train_x[106:140]

library(ggplot2)
tsData = rbind(
     data.frame(seq = 1:length(train_x[1:35]), data = train_x[1:35], cls = 'x1')
    , data.frame(seq = 1:length(train_x[36:70]), data = train_x[36:70], cls = 'x2')
    , data.frame(seq = 1:length(train_x[71:105]), data = train_x[71:105], cls = 'x3')
    , data.frame(seq = 1:length(train_x[106:140]), data = train_x[106:140], cls = 'x4')
    , data.frame(seq = 1:length(train_x[141:175]), data = train_x[141:175], cls = 'x5'))

png("sp500-features.png", height = 800, width = 600)
ggplot(tsData, aes(seq, data)) + geom_line(aes(colour = cls))
dev.off()
system('explorer sp500-features.png')
tsData = rbind(
     data.frame(seq = 1:length(train_y[1:35]), data = train_x[1:35], cls = 'y1')
    , data.frame(seq = 1:length(train_y[36:70]), data = train_x[36:70], cls = 'y2')
    , data.frame(seq = 1:length(train_y[71:105]), data = train_x[71:105], cls = 'y3')
    , data.frame(seq = 1:length(train_y[106:140]), data = train_x[106:140], cls = 'y4')
    , data.frame(seq = 1:length(train_y[141:175]), data = train_x[141:175], cls = 'y5'))
png("sp500-labels.png")
ggplot(tsData, aes(seq, data)) + geom_line(aes(colour = cls))
dev.off()
system('explorer sp500-labels.png')