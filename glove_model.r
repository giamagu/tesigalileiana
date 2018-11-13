library(keras)

#alleno i word embeddings sulle recensioni del database IMDB
#considero solo 2800 parole e mi fermo a leggere la recensione dopo 50 parole
max_features <- 2800
maxlen <- 35
n_classes <- 46
dim_embedding <- 48
window_size <- 8
negative_samples <- 9
ndocs <- 8982

#scarico il dataset
reuters <- dataset_reuters(num_words = max_features)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% reuters

x_train_complete <- pad_sequences(train_data, maxlen = maxlen)
y_train <- train_labels
x_test <- pad_sequences(test_data, maxlen = maxlen)
y_test <- test_labels

#ho copiato dal libro questo passaggio. Dovrebbe ritornare dei dizionari che associano una parola al corrispondente intero e viceversa
word_index <- dataset_reuters_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index

#costruisco la matrice di cooccorenza che serve in GloVe
co_occ_matrix <- list()

#bisogna vedere bene come funziona tokenizer, per capire che numero corrisponde ai caratteri speciali e controllare quella cosa che ogni tanto con max_features va il +1 e ogni tanto no
#poi si può creare un dizionario da zero, non pare ci siano funzioni a disposizione in R

#non lavoro con tutti i documenti allo stesso tempo per questioni di memoria. Questo non va benissimo, ma facciamo finta di niente, almeno per ora.
dim_batch <- 250
for (i in 0:(ndocs/dim_batch - 1)){
	couples <- c()
	for (index in(i*dim_batch+1): ((i+1)*dim_batch)){
		doc <- x_train_complete[index,]
		#skipgrams usa finestre che vanno avanti e indietro. Non dovrebbe essere cosi' in GloVe, ma non dovrebbe cambiare niente
		doc_couples <- skipgrams(doc, max_features, window_size = window_size, negative_samples = 0)[1]$couples
		
		#creo la matrice di co-occorrenza dei documenti, costruita come lista di liste
		for(couple in doc_couples){
			#se la prima parola e' gia' stata inizializzata mi concentro sulla seconda parola
			if (couple[1] %in% names(co_occ_matrix)){
				#se anche la seconda parola è stata trovata, sommo uno
				if (couple[2] %in% names(co_occ_matrix[[toString(couple[1])]])){
					co_occ_matrix[[toString(couple[1])]][[toString(couple[2])]] <- co_occ_matrix[[toString(couple[1])]][[toString(couple[2])]] + 1
					}
				#altrimenti inizializzo la seconda parola
				else{
					co_occ_matrix[[toString(couple[1])]] [[toString(couple[2])]] <- 1
					}
				}
			#se la prima parola viene trovata per la prima volta inizializzo la lista
			else{
				co_occ_matrix[[toString(couple[1])]] <- list()
				co_occ_matrix[[toString(couple[1])]] [[toString(couple[2])]] <- 1
				}
			}
		}
	}
	
	
	
#creo il modello

#input
context_input <- layer_input(shape = 1, dtype = "int32", name = "context")
target_input <- layer_input(shape = 1, dtype = "int32", name = "target")
weight_input <- layer_input(shape = 1, dtype = "float32", name = "weight")

#creo i due vettori di cui fare il prodotto scalare
context_embedding <- context_input %>%
	layer_embedding(input_dim = max_features+1, output_dim = dim_embedding)
	
target_embedding <- target_input %>%
	layer_embedding(input_dim = max_features+1, output_dim = dim_embedding)
	
#come output uso il prodotto scalare, moltiplicato per il peso
score <- layer_dot(list(context_embedding, target_embedding), axes = c(2,2))%>%
	layer_flatten()
output <- layer_multiply(list(score, weight_input))

glove_model <- keras_model(list(context_input, target_input, weight_input), output)

adam <- optimizer_adam(lr = 0.0001)

glove_model %>% compile(
	optimizer = adam,
	loss = "mse")
		
#alleno GloVe

iter <- 0
#per ogni documento prendo a caso queste parole
sampled_words <- 2000
while(iter < 50){
	#inizializzo i vettori di input
	context_words <- c(300*sampled_words)
	target_words <- c(300*sampled_words)
	log_frequencies <- numeric(300*sampled_words)
	weight_frequences <- numeric(300*sampled_words)
	len_words <- 0
	for (extraction in 0:299){
		#scelgo un documento a caso da cui estrarre sampled_words coppie
		document <- x_train_complete[floor(runif(1, 1, ndocs + 0.99999)),]
		couples <- skipgrams(document, max_features, window_size = window_size, negative_samples = 0)[1]$couples
		len_couples <- length(couples)
		doc_context_words <- numeric(len_couples)
		doc_target_words <- numeric(len_couples)
		for (i in 1:len_couples){
			len_words <- len_words + 1
			context_words[len_words] <- couples[[i]][1]
			target_words[len_words] <- couples[[i]][2]
			}
		}
		print(iter)
	context_words <- context_words[1:len_words]
	target_words <- target_words[1:len_words]
	log_frequencies <- log_frequencies[1:len_words]
	weight_frequences <- weight_frequences[1:len_words]
	for (i in 1:(len_words)){
		first_word <- context_words[i]
		second_word <- target_words[i]
		frequence <- co_occ_matrix[[toString(first_word)]][[toString(second_word)]]
		#se per caso scelgo una coppia di parole mai viste (ma non dovrebbe succedere, a meno di problemi ai bordi), metto la frequenza a zero e l'errore non conta piu' molto. x log(x) va a zero con x che va a zero.
		if (length(frequence) < 1){
			frequence <- 0.00001
			print(i)
		}
		#modulo la frequenza come si fa con GloVe. La mia frequenza e' la radice di quella di Glove, viene elevata da mse.
		if (frequence < 70){
			weight_frequence <- frequence^(0.375)
			}
		else{
			weight_frequence <- 70^(0.375)
			}
		weight_frequences[i] <- weight_frequence
		log_frequencies[i] <- log(frequence)*weight_frequence
		}
	glove_model %>% fit(list(target_words, context_words, weight_frequences), log_frequencies, 
		epochs = 1, batch_size = 512)
	iter <- iter + 1
	}

#creo i dizionari

final_embedding <- layer_add(list(context_embedding, target_embedding))
dictionary <- keras_model(list(context_input, target_input), final_embedding)

#testo la cos similarity

#vero e proprio test: cerco le 15 parole più vicine a una parola obiettivo, che chiamo center.
index <- word_index[["oil"]]
#center e' l'embedding della parola corrispondente. Per fare il prodotto scalare deve avere le giuste dimensioni
center <- dictionary %>% predict(list(index, index))
center <- as.matrix(center)
dim(center) <- c(dim_embedding,1)
#creo due array che contengono le parole vicine e le relative distanze. Inizializzo solo similarities perche' sara' utile. Potevo inizializzarlo sicuramente in modo piu' compatto, ma ogni tanto sono pigro
closest <- c()
closest_encoded <- c()
similarities <- c(-10000, -10000, -10000, -10000, -10000,-10000, -10000, -10000, -10000, -10000,-10000, -10000, -10000, -10000, -10000)
#il ciclo è troppo lungo, finisce con un errore. Credo di doverlo bloccare al numero di parole nel dizionario, ma mi sento più sicuro a farlo andare in errore alla fine
for (encoded_word in 1:(max_features-4)){
	#creo il candidato ad essere vicino al centro
	candidate <- dictionary %>% predict(list(encoded_word, encoded_word))
	candidate <- as.matrix(candidate)
	#vedo se il candidato e' piu' vicino delle attuali 15 componenti di closest
	#se lo e', i sara' al massimo 15
	i <- 1
	while(i<16 & (t(center) %*% candidate)/(norm(candidate,"f")*norm(center,"f")) < similarities[i]){
		i <- i+1
		}
	#distinguo i due casi perche' temo che con R avrei dei problemi a non farlo
	#inserisco il candidato nell'i-esimo posto, mantenendo le due liste, closest e similarities, ordinate
	if (i < 15){
		#sposto la parte destra dell'array di una posizione
		similarities[(i+1):15] <- similarities[i:14]
		#calcolo la similarity e la inserisco nella i-esima posizione
		similarities[i] <- (t(center) %*% candidate)/(norm(candidate,"f")*norm(center,"f"))
		#se i fosse 15 questa istruzione sarebbe problematica in R.
		closest[(i+1):15] <- closest[i:14]
		closest[i] <- reverse_word_index[as.character(encoded_word)]
		closest_encoded[(i+1):15] <- closest_encoded[i:14]
		closest_encoded[i] <- encoded_word
		}
	if (i == 15){
		similarities[i] <- (t(center) %*% candidate)/(norm(candidate,"f")*norm(center,"f"))
		closest[i] <- reverse_word_index[as.character(encoded_word)]
		closest_encoded[i] <- encoded_word
		}
	}
print(closest)
print(index)
print(closest_encoded)