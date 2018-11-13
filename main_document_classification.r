my_classifications <- numeric(n_classes)
correct_classifications <- numeric(n_classes)
total_classifications <- numeric(n_classes)
class_matrix <- matrix(numeric(n_classes*n_classes), nrow = n_classes, ncol = n_classes)
correct_classifications_top_3 <- numeric(n_classes)

freq_topics = numeric(46)
for (topic in y_train){
    freq_topics[topic + 1] <- freq_topics[topic + 1] + 1
}
log_freq_topics <- log(freq_topics)

for (n in 1:1000){
	#riga per la classificazione su otto classi
	if ((y_test[n]+1) %in% c(2,4,5,12,14,17,20,21)){
		print(n)
		document <- test_data[[n]]
		log_likelihood <- numeric(n_classes)
		c(input_couples, input_labels) %<-% skipgrams(document, max_features, window_size=window_size, negative_samples=0, shuffle = FALSE)
		document_context <- numeric(2 * window_size * len)
		document_target <- numeric(2 * window_size * len)
		constant <- numeric(2 * window_size * len)
		j <- 0
		for (couple in input_couples){
			first_word_exists <- !(couple[1] < 7)
			second_word_exists <- !(couple[2] < 7)
			if (first_word_exists & second_word_exists){
				j <- j + 1
				document_context[j] = couple[1]
				document_target[j] = couple[2]
				constant[j] <- 1
				}
			}
		document_context <- document_context[1:j]
		document_target <- document_target[1:j]
		constant <- constant[1:j]
		
		if(j > 0){
			#riga per la classificazione su otto classi
			for (topic in c(2,4,5,12,14,17,20,21)){
			#for (topic in 1:n_classes){
				topic_vectors <- array(numeric(j*n_classes), dim = c(j,n_classes))
				for (k in 1:j){
				topic_vectors[k, topic] <- 1
					}
				probs <- model %>% predict(list(document_target, document_context, constant, topic_vectors))
				for (prob in probs){
					log_likelihood[topic] <- log_likelihood[topic] + log(prob)
					}
				log_likelihood[topic] <- log_likelihood[topic] + 28 * log_freq_topics[topic]
				}
			
			#classe singola
			predicted_class <- 0
			max_class <- -9999999999999999999999999
			#riga per la classificazione su otto classi
			for (topic in c(2,4,5,12,14,17,20,21)){
			#for (topic in 1:n_classes){
				if (log_likelihood[topic] > max_class){
					max_class <- log_likelihood[topic]
					predicted_class <- topic
					}
				}
			class_matrix[predicted_class, y_test[n]+1] = class_matrix[predicted_class, y_test[n]+1] + 1
			if (predicted_class == y_test[n]+1){
				correct_classifications[y_test[n]+1] <- correct_classifications[y_test[n]+1]+ 1
				}
			total_classifications[y_test[n]+1] <- total_classifications[y_test[n]+1] + 1
			my_classifications[predicted_class] <- my_classifications[predicted_class] + 1
			
			#hit top 3
			predicted_classes <- c(0, 0, 0)
			max_classes <- c(-9999999999999999999999999,-9999999999999999999999999,-9999999999999999999999999)
			for (topic in c(2,4,5,12,14,17,20,21)){
				if (log_likelihood[topic] > max_classes[1]){
					max_classes[2:3] <- max_classes[1:2]
					max_classes[1] <- log_likelihood[topic]
					predicted_classes[2:3] <- predicted_classes[1:2]
					predicted_classes[1] <- topic
					}
				else if (log_likelihood[topic] > max_classes[2]){
					max_classes[3] <- max_classes[2]
					max_classes[2] <- log_likelihood[topic]
					predicted_classes[3] <- predicted_classes[2]
					predicted_classes[2] <- topic
					}
				else if (log_likelihood[topic] > max_classes[3]){
					max_classes[3] <- log_likelihood[topic]
					predicted_classes[3] <- topic
					}
				}
			if ((y_test[n]+1) %in% predicted_classes){
				correct_classifications_top_3[y_test[n]+1] <- correct_classifications_top_3[y_test[n]+1] + 1
				}
			}
		}
	}
# print(correct_classifications[c(2,4,5,12,14,17,20,21)])
# print(correct_classifications_top_3[c(2,4,5,12,14,17,20,21)])
# print(total_classifications[c(2,4,5,12,14,17,20,21)])
# print(my_classifications[c(2,4,5,12,14,17,20,21)])
# print(sum(correct_classifications[c(2,4,5,12,14,17,20,21)]))
# print(sum(correct_classifications_top_3[c(2,4,5,12,14,17,20,21)]))
# print(sum(total_classifications[c(2,4,5,12,14,17,20,21)]))
# print(sum(my_classifications[c(2,4,5,12,14,17,20,21)]))
# print(class_matrix[c(2,4,5,12,14,17,20,21),c(2,4,5,12,14,17,20,21)])
