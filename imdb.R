library(keras)

rm(list = ls())

imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

str(train_data[[1]])
max(sapply(train_data, max))

# 0 = negative, 1 = positive
str(train_labels)

word_index <- dataset_imdb_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index

# Decodes the review. Note that the indices are offset by 3 because 0, 1, and 
# 2 are reserved indices for “padding,” “start of sequence,”and “unknown.”

# Original
decoded_review <- sapply(train_data[[1]], function(index) {
  word <- if (index >= 3) 
    reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) 
    word
  else 
    "?"
})
paste(decoded_review, collapse = " ")

# Alternative
# decoded_review_2 <- sapply(train_data[[1]] - 3, function(index) {
#   word <- if (index > 0) 
#     reverse_word_index[[as.character(index)]]
#   else 
#     "?"
# })
# paste(decoded_review_2, collapse = " ")
# 
# identical(decoded_review, decoded_review_2)

# vectorize_sequences <- function(sequences, dimension = 10000) {
#   results <- matrix(0, nrow = length(sequences), ncol = dimension)
#   for (i in 1:length(sequences))
#     results[i, sequences[[i]]] <- if (sequences[[i]] >= 3)
#       reverse_word_index[[as.character(sequences[[i]] - 3)]]
#   else
#     "?"
#   results
# }

# train1 <- vectorize_sequences(train_data[[1]])
# train1[1:30, 1:30]

vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

val_indices <- 1:10000
x_val <- x_train[val_indices, ]
partial_x_train <- x_train[-val_indices, ]

y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

# 1st attempt - 10 epochs
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = c("accuracy"))

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 10,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

# plot(history)
history_df <- as.data.frame(history)
# View(history_df)

# 2nd attempt - 4 epochs - same data
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = c("accuracy"))
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 4,
  batch_size = 512
)

results <- model %>% evaluate(x_val, y_val)
results$loss
results$acc

# Predicting
predict(model, x_test[1:10, ])

decoded_review <- sapply(test_data[[2]], function(index) {
  word <- if (index >= 3) 
    reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) 
    word
  else 
    "?"
})
paste(decoded_review, collapse = " ")

decoded_review <- sapply(test_data[[8]], function(index) {
  word <- if (index >= 3) 
    reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) 
    word
  else 
    "?"
})
paste(decoded_review, collapse = " ")

# Further attempts
# 3
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = c("accuracy"))

history <- model %>% fit(
  x_train,
  y_train,
  epochs = 5,
  batch_size = 512,
  validation_data = list(x_test, y_test)
)

results <- model %>% evaluate(x_test, y_test)
results$loss
results$acc
