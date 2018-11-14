library(keras)

# digits

# from https://tensorflow.rstudio.com/keras/

mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Backup up
x_train_o <- mnist$train$x
y_train_o <- mnist$train$y
x_test_o <- mnist$test$x
y_test_o <- mnist$test$y

image(matrix(x_train_o[181, 1:24, 1:24], nrow = 24, ncol = 24))
y_train_o[181]

x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

model %>% evaluate(x_test, y_test)

prediction <- model %>% predict_classes(x_test)

image(matrix(x_test_o[2, 1:24, 1:24], nrow = 24, ncol = 24))
y_test_o[2]
prediction[2]

image(matrix(x_test_o[12, 1:24, 1:24], nrow = 24, ncol = 24))
y_test_o[12]
prediction[12]

df <- data.frame(cbind(test = y_test_o, 
                       pred = prediction, 
                       ident = (y_test_o == prediction)))
correct <- sum(df$ident)
incorrect <- sum(!df$ident)
score <- correct / (correct + incorrect)

View(df[df$ident == FALSE, ])

image(matrix(x_test_o[449, 1:24, 1:24], nrow = 24, ncol = 24))
y_test_o[449]
prediction[449]

image(matrix(x_test_o[685, 1:24, 1:24], nrow = 24, ncol = 24))
y_test_o[685]
prediction[685]

image(matrix(x_test_o[341, 1:24, 1:24], nrow = 24, ncol = 24))
y_test_o[341]
prediction[341]

image(matrix(x_test_o[1329, 1:24, 1:24], nrow = 24, ncol = 24))
y_test_o[1329]
prediction[1329]

# fashion

# https://tensorflow.rstudio.com/keras/articles/tutorial_basic_classification.html


