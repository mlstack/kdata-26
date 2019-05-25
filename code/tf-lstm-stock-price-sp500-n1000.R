#!/usr/bin/env /c/Apps/Anaconda3/Scripts/Rscript

# forecasting price
# modified
# Author : Yibeck Lee(yibeck.lee@gmail.com)
# references 
#  1) https://github.com/BinRoot/TensorFlow-Book/
#  2) https://github.com/haven-jeon/TensorFlow-Book-R
#  3) https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
# sess = tf$InteractiveSession()
# a = tf$constant(5.0)
# b = tf$constant(6.0)
# c = a * b
# We can just use 'c.eval()' without passing 'sess'
# print(c$eval())
# sess$close()

library(tensorflow)
library(ggplot2)
options(tensorflow.one_based_extract = FALSE)
fileName = 'sp500-n1000.csv'
data = read.csv(fileName, header = F)[1:1000, 1]
#tsData = rbind(
#     data.frame(seq = 1:length(data), data = data, cls = 'sp500'))
# ggplot(tsData, aes(seq, data)) + geom_line(aes(colour = cls))
# data = seq(1, 100, 1)
# data

# input 변수 표준화 : mean = 0, std = 1
data = scale(data)

# 관찰치수
numObs = length(data)

# trainData[시작시간 ~ 종료*0.8], testData[종료*0.8 ~ 종료]
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



# 모델(y = Wx + b)의 변수 초기값 할당 - Weight, bias
W = tf$Variable(tf$random_normal(list(hiddenDim, 1L)), name = 'W')
b = tf$Variable(tf$random_normal(list(1L)), name = 'bias')
# 외부데이터 저장공간 - placeholder 
x = tf$placeholder(dtype = tf$float32, shape = list(NULL, seqLength, inputDim), name='feature')
y = tf$placeholder(dtype = tf$float32, shape = list(NULL, seqLength), name='label')
# RNN 프로세스 방식 : Basic LSTM recurrent network cell 
# cell = tf$contrib$rnn$BasicLSTMCell(hiddenDim)
cell = tf$contrib$rnn$LSTMCell(
     num_units = hiddenDim
    , forget_bias = 1.0 # bias of forget gete to reduce the scale of forgetting in the beginning of the training
    , activation = tf$nn$tanh # 활성화 함수 : hyper tangent  [, relu, ...]
    )

# dynimic rnn : Performs fully dynamic unrolling of inputs
outputs_states = tf$nn$dynamic_rnn(cell, x, dtype = tf$float32)
num_examples = tf$shape(x)[0]
# tf.tile(input, multiples, name = None)
W_repeated = tf$tile(tf$expand_dims(W, 0L), list(num_examples, 1L, 1L))

# model y = Wx + b
modelOut = tf$matmul(outputs_states[[1]], W_repeated) + b
modelOut = tf$squeeze(modelOut)

# mse(mean squared error) of [model - label]
cost = tf$reduce_mean(tf$square(modelOut - y))

# optimizer : Adam algorithm [,GradientDescent, ...] 
train_op = tf$train$AdamOptimizer()$minimize(cost)

# movel saver 할당 : local repository에 binary model file 생성
saver = tf$train$Saver()

with(tf$Session() %as% sess, {
    tf$get_variable_scope()$reuse_variables()
    sess$run(tf$global_variables_initializer()) # 변수 초기화
#    for (i in 1:100) {
#        mse_ = sess$run(list(train_op, cost), feed_dict = dict(x = train_x, y = train_y))
#        if (i %% 100) {
#            print(paste(i, mse_[[2]]))
#        }
#        save_path = saver$save(sess, 'c:/Home/GitHub/e200-tf/model/model.ckpt')
#        print(sprintf('Model saved to %s', save_path))
#    }

    # 최대 허용치
    max_patience = 3 
    patience = max_patience
    # 학습된 모델식에 testData를 적용할 때의 최소 오차
    min_test_err = Inf # infite : 무한대
    step = 0
    # patient가 0이 될대까지 진행
    while (patience > 0) {
        NULL_train_err = sess$run(list(train_op, cost),
                                  feed_dict = dict(x = train_x, y = train_y))
        train_err = NULL_train_err[[2]]
        if (step %% 100 == 0) {
        test_err = sess$run(cost, feed_dict = dict(x = test_x, y = test_y))
        cat(sprintf('step: %d\t\ttrain err: %f\t\ttest err: %f\n', step, train_err, test_err))
        if (test_err < min_test_err) {
            min_test_err = test_err
            patience = max_patience
            } else {
                patience = patience - 1
                }
            }
        step = step + 1
    }
    # 모델 binary file 저장 path
    save_path = saver$save(sess, './model.ckpt')
    print(sprintf('Model saved to %s', save_path))

    tf$get_variable_scope()$reuse_variables()
    saver$restore(sess, './model.ckpt')
    predicted_vals = sess$run(modelOut, feed_dict = dict(x = test_x))
    print(sprintf('predicted_vals, %s', dim(predicted_vals)))

    tsData = rbind(
     data.frame(seq = 1:length(trainData), data = trainData, cls = 'training')
    , data.frame(seq = length(trainData):(length(trainData) + length(predicted_vals[, 1]) - 1),
               data = predicted_vals[, 1], cls = 'forecasted')
    , data.frame(seq = length(trainData):(length(trainData) + length(testData) - 1),
               data = testData, cls = 'test-actual'))
    print(ggplot(tsData, aes(seq, data)) + geom_line(aes(colour = cls)))

    # 예측값 생성
    prev_seq = train_x[nrow(train_x),,]
    predicted_vals = c()
    for (i in 1:20) {
        tf$get_variable_scope()$reuse_variables()
        saver$restore(sess, './model.ckpt')
        next_seq = sess$run(modelOut, feed_dict = dict(x = array(prev_seq, dim = c(1, 5, 1))))
        predicted_vals = c(predicted_vals, next_seq[5])
        prev_seq = c(prev_seq[2:5], next_seq[5])
    }

    tsData = rbind(
     data.frame(seq = 1:length(trainData), data = trainData, cls = 'training')
    , data.frame(seq = length(trainData):(length(trainData) + length(predicted_vals) - 1),
               data = predicted_vals, cls = 'forecasted')
    , data.frame(seq = length(trainData):(length(trainData) + length(testData) - 1),
               data = testData, cls = 'test-actual'))
    print(ggplot(tsData, aes(seq, data)) + geom_line(aes(colour = cls)))
})
