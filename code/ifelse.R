data<-c(0,1,2,3,4,2,3,1,4,3,2,4,0,1,2,0,2,1,2,0,4)
frame<-as.data.frame(data)
frame$L_1 <- ifelse(frame$data>=2, 2, 1)
frame$L_2 <- ifelse(frame$data>=2, 2, 1)
frame$L_3 <- ifelse(frame$data>=2, 2, 1)

print(frame)
