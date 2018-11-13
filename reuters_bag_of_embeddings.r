library(keras)
library(rlist)

max_features <- 2800
maxlen <- 300
n_classes <- 46
dim_embedding <- 48
window_size <- 8
negative_samples <- 9

skipgram_model <- load_model_hdf5("skipgram_reuters_model.h5")

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

#creo il modello

#creo per prima cosa il layer per l'embedding  del contesto e lo restringo a vettori di dimensione dim_embedding
context_input <- layer_input(shape = 1, dtype = "int32", name = "context")

context_embedding_layer <- layer_embedding(input_dim = max_features, output_dim = dim_embedding)

encoded_context <-  context_input %>%
	context_embedding_layer
	
#faccio lo stesso per il target. Forse era il caso di far partire il target da 0 invece di rendere random anche gli embeddings iniziali della parole target?
target_input <- layer_input(shape = 1, dtype = "int32", name = "target")

target_embedding_layer <- layer_embedding(input_dim = max_features, output_dim = dim_embedding*n_classes)
	
possible_encoded_targets <- target_input %>%
	target_embedding_layer %>%
	layer_reshape(target = c(n_classes, dim_embedding)) %>%
	layer_permute(dims = c(2,1))
	
#USO QUI INPUT_LABELS
pos_neg_input <- layer_input(shape = 1, dtype = "float32", name = "labels")

#Qui inserisco la classe del documento
class_input <- layer_input(shape = n_classes, dtype = "float32", name = "class")

#Qui scelgo la giusta classe per il target
encoded_target <- layer_dot(list(possible_encoded_targets,class_input), axes = c(2,1))

#a questo punto faccio il prodotto scalare dei due embeddings e applico la sigmoid
scalar_product <- layer_dot(list(encoded_context, encoded_target), axes = c(2,1))

output_entropy <- layer_dot(list(scalar_product,pos_neg_input), axes = 1)%>%
	layer_activation(activation = 'sigmoid') %>%
	layer_reshape(target = c(1))
	
output_probability <- layer_activation(scalar_product, activation = 'sigmoid') %>%
	layer_reshape(target = c(1))

#creo e compilo il modello usando la binary crossentropy come loss function
model <- keras_model(list(context_input, target_input, pos_neg_input, class_input), output_entropy)

optimizer <- optimizer_adam(lr = 0.0001)
model %>% compile(
optimizer = optimizer,
loss = "binary_crossentropy"
)

#fare attenzione che skipgram e BoE hanno target e context invertiti
layers <- skipgram_model$layers
context_weights <- get_weights(layers[[3]])
target_weights <- get_weights(layers[[4]])
target_matrix <- matrix(numeric(dim_embedding*n_classes*max_features), nrow = max_features, ncol = n_classes*dim_embedding)
for (i in 1:n_classes){
	target_matrix[1:2800, ((i-1)*dim_embedding + 1): (i*dim_embedding)] <- target_weights[[1]]
	}
target_matrix <- list(target_matrix)
set_weights(model$layers[[7]], context_weights)
set_weights(model$layers[[2]], target_matrix)

#creo il modello che restituisce la probabilita'
prob_model <- keras_model(list(context_input, target_input, class_input), output_probability)

optimizer <- optimizer_adam(lr = 0.0005)

n_rec <- 60

#questo parametro misura la dimesione degli input in ogni ciclo. Serve per inizializzare vettori della giusta dimensione, e quindi per avere molta piu' velocita'
len <- 2*(maxlen*window_size - window_size*(window_size+1)/2) * (negative_samples+1)

#IL MODELLO E' MOLTO LENTO PERCHE' GIRA SU TUTTE LE RECENSIONI FACENDO 2 EPOCHE. INOLTRE LA WINDOW SIZE, I NEGATIVE SAMPLINGS E LE DIMENSIONI DEGLI EMBEDDING SONO GRANDI
target_embedding_layer$trainable <- TRUE
context_embedding_layer$trainable <- FALSE

model %>% compile(
optimizer = optimizer,
loss = "binary_crossentropy"
)

memory_index <- numeric(n_classes)
for (l in 0:40){
	print(l)
	x_train <- list()
	upper_document_class <- numeric(n_rec)
	for (topic in 1:n_classes){
		last_document <- memory_index[topic] + 1
		while(y_train[last_document] != (topic - 1)){
			last_document <- last_document %% 8800
			last_document <- last_document + 1
			}
		memory_index[topic] <- last_document
		x_train <- list.append(x_train, train_data[[last_document]])
		upper_document_class[topic] <- y_train[last_document] + 1		#il +1 serve perche' y_train e' fatto di zeri e uni. La classe 0 e' rappresentata dal vettore e_1 della base canonica, la classe 2 dal vettore e_2
																	#prendo la i-esima colonna della matrice, che ha soli zeri, e cambio una delle entrate. Importante che y_train ha tutte le recensioni!!
		}
	for (remaining in (n_classes + 1):n_rec){
		random_number <- floor(runif(1, 1, 8800.999))
		x_train <- list.append(x_train, train_data[[random_number]])
		upper_document_class[remaining] <- y_train[random_number]+1		#il +1 serve perche' y_train e' fatto di zeri e uni. La classe 0 e' rappresentata dal vettore e_1 della base canonica, la classe 2 dal vettore e_2
															#prendo la i-esima colonna della matrice, che ha soli zeri, e cambio una delle entrate. Importante che y_train ha tutte le recensioni!!
		}
	#divido context e target in due array distinti
	context_words <- numeric(len*n_rec)
	target_words <- numeric(len*n_rec)
	input_labels <- numeric(len*n_rec)
	#creo una matrice per tenere in memoria la classe dei documenti, a ogni classe associo un vettore della base canonica.
	#Suppongo che in input le classi dei documenti siano rappresentate da numeri interi e di sapere a priori quante sono le classi.
	document_class <- matrix(0,n_classes,1.1*len*n_rec)

	
	i <- 0
	for (n in 1:n_rec){
		document <- x_train[[n]]
		topic_class <- upper_document_class[n]
		#la funzione skipgrams ritorna coppie di parole del tipo (context,target), con 6 negative samples per ogni coppia "giusta"
		#inoltre crea dei label che indicano se la coppia era giusta oppure no
		#i negative samples li cambiero' io a mano piu' avanti nel codice.
		c(input_couples, local_input_labels) %<-% skipgrams(document, max_features, window_size=window_size, negative_samples=negative_samples)
		
		j <- 0
		for (couple in input_couples){
			j <- j+1
			#i numeri 1, 2, 3 nei documenti rappresentano cose non interessanti, quindi elimino tutte le coppie che li contengono.
			if (!(couple[1] %in% c(1,2,3))){
				#mantengo pero' le coppie con 1, 2, 3 generati da skipgrams, li sostituisco io piu' avanti
				if (!(couple[2] %in% c(1,2,3) & local_input_labels[j] == 1)){
					i <- i+1
					#la distribuzione di skiprams fa schifo,
					#sostituisco i random samples con una distribuzione che creo io. Se non ci sono sostituzioni da fare, il sample sara' quello restiuito da skipgrams.
					#La distribuzione e' fatta in modo da approssimare la legge di Zipf.
					random_sample <- couple[2]
					if (local_input_labels[j] == 0){
						random_sample <- max_features
						while (random_sample>max_features-1){					#potenzialmente estraggo termini troppo grandi. Se succede, riprovo.
							randvar <- runif(1,0.0000001,1)
							random_sample <- floor(1/randvar^14.5)+3	#il +3 dipende sempre dal fatto che non voglio i primi tre interi.
							}
						}
					context_words[i] <- random_sample		#random_sample coincide con couple[2] se non e' un negative sampling
					target_words[i] <- couple[1]
					document_class[topic_class,i] <- 1		#il + 1 e' gia' contato
					#input_labels deve essere -1 o +1, non 0 o 1. Serve per calcolare bene la loss function.
					input_labels[i] <- -1+2*local_input_labels[j]
					}
				}
			}
		}


	#qui controllo se per caso una recensione aveva meno di 50 parole. In tal caso, accorcio i vettori di input per non avere zeri alla fine. Lo faccio controllando che i sia effettivamente cio' che dovrebbe
	if (i!=len*n_rec){
		context_words <- context_words[1:i]
		target_words <- target_words[1:i]
		input_labels <- input_labels[1:i]
		document_class <- document_class[,1:i]
		}
	document_class <- t(document_class)
	one_vector <- rep(1, i)
	
	#alleno il modello usando come output input_labels, che sarebbe l'array di zeri e uni creato da skiprams.
	#l'allenamento del modello lo faccio per una sola epoca. Visto che la parte piu' costosa e' creare gli input, vale la pena inserire questo passaggio in un ciclo.
	#Non lo faccio in fase sperimentale perche' comunque richiede tempo
	model %>% fit(
	list(context_words, target_words, input_labels, document_class), one_vector,
	epochs = 1, batch_size = 256
	)
	#fine allenamento
	}

	
#dizionari
dictionary_context <- keras_model(context_input, encoded_context)
dictionary_target <- keras_model(list(target_input,class_input), encoded_target)


model_test(dictionary_context, word_index, max_features, dim_embedding)


#TEST PER IL CONTESTO
#vero e proprio test: cerco le 15 parole più vicine a una parola obiettivo, che chiamo center.
index <- word_index["bad"]
#center e' l'embedding della parola corrispondente. Per fare il prodotto scalare deve avere le giuste dimensioni
center <- dictionary_context %>% predict(as.numeric(index)+3)
center <- as.matrix(center)
dim(center) <- c(dim_embedding,1)
#creo due array che contengono le parole vicine e le relative distanze. Inizializzo solo similarities perche' sara' utile. Potevo inizializzarlo sicuramente in modo piu' compatto, ma ogni tanto sono pigro
closest <- c()
closest_encoded <- c()
similarities <- c(-10000, -10000, -10000, -10000, -10000,-10000, -10000, -10000, -10000, -10000,-10000, -10000, -10000, -10000, -10000)
#il ciclo è troppo lungo, finisce con un errore. Credo di doverlo bloccare al numero di parole nel dizionario, ma mi sento più sicuro a farlo andare in errore alla fine
for (encoded_word in 1:(max_features-4)){
	#creo il candidato ad essere vicino al centro
	candidate <- dictionary_context %>% predict(encoded_word+3)
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
	
#TEST PER IL TARGET
#vero e proprio test: cerco le 15 parole più vicine a una parola obiettivo, che chiamo center.
index <- word_index["this"]
#per cambiare classe, invertire 0 e 1.
class_vector <- t(c(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))
#center e' l'embedding della parola corrispondente. Per fare il prodotto scalare deve avere le giuste dimensioni
center <- dictionary_target %>% predict(list(as.numeric(index)+3,class_vector))
center <- as.matrix(center)
#creo due array che contengono le parole vicine e le relative distanze. Inizializzo solo similarities perche' sara' utile. Potevo inizializzarlo sicuramente in modo piu' compatto, ma ogni tanto sono pigro
closest <- c()
closest_encoded <- c()
similarities <- c(-10000, -10000, -10000, -10000, -10000,-10000, -10000, -10000, -10000, -10000,-10000, -10000, -10000, -10000, -10000)
#il ciclo è troppo lungo, finisce con un errore. Credo di doverlo bloccare al numero di parole nel dizionario, ma mi sento più sicuro a farlo andare in errore alla fine
for (encoded_word in 1:max_features-4){
	#creo il candidato ad essere vicino al centro
	candidate <- dictionary_target %>% predict(list(as.numeric(encoded_word)+3,class_vector))
	candidate <- as.matrix(candidate)
	#vedo se il candidato e' piu' vicino delle attuali 15 componenti di closest
	#se lo e', i sara' al massimo 15
	i <- 1
	sim <- (candidate %*% t(center))/(norm(candidate,"f")*norm(center,"f"))
	# if (sim < similarities[15]){
		# i <- 16
		# }
	while(i<16 &  sim < similarities[i]){
		i <- i+1
		}
	#distinguo i due casi perche' temo che con R avrei dei problemi a non farlo
	#inserisco il candidato nell'i-esimo posto, mantenendo le due liste, closest e similarities, ordinate
	if (i < 15){
		#sposto la parte destra dell'array di una posizione
		similarities[(i+1):15] <- similarities[i:14]
		#calcolo la similarity e la inserisco nella i-esima posizione
		similarities[i] <- sim
		#se i fosse 15 questa istruzione sarebbe problematica in R.
		closest[(i+1):15] <- closest[i:14]
		closest[i] <- reverse_word_index[as.character(encoded_word)]
		closest_encoded[(i+1):15] <- closest_encoded[i:14]
		closest_encoded[i] <- encoded_word
		}
	if (i == 15){
		similarities[i] <- sim
		closest[i] <- reverse_word_index[as.character(encoded_word)]
		closest_encoded[i] <- encoded_word
		}
	}
print(closest)
print(index)
print(closest_encoded)


model %>% save_model_hdf5("final_reuters_model.h5")
prediction_model %>% save_model_hdf5("prediction_final_reuters_model.h5")

model <- load_model_hdf5("final_reuters_model.h5")
prediction_model <- load_model_hdf5("prediction_final_reuters_model.h5")