library(keras)

#alleno i word embeddings sulle recensioni del database IMDB
#considero solo 2800 parole e mi fermo a leggere la recensione dopo 50 parole
max_features <- 2800
maxlen <- 350
n_classes <- 46
dim_embedding <- 48
window_size <- 8
negative_samples <- 9

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
#creo per prima cosa il layer per l'embedding  del contesto e lo restringo a vettori di dimensione 32 (forse 32 e' troppo)
context_input <- layer_input(shape = 1, dtype = "int32", name = "context")

encoded_context <-  context_input %>%
	layer_embedding(input_dim = max_features, output_dim = dim_embedding)
	
#faccio lo stesso per il target. Forse era il caso di far partire il target da 0 invece di rendere random anche gli embeddings iniziali della parole target?
target_input <- layer_input(shape = 1, dtype = "int32", name = "target")

encoded_target <-  target_input %>%
	layer_embedding(input_dim = max_features, output_dim = dim_embedding)
	
#USO QUI INPUT_LABELS
pos_neg_input <- layer_input(shape = 1, dtype = "float32", name = "labels")
	
#a questo punto faccio il prodotto scalare dei due embeddings e applico la sigmoid
scalar_product <- layer_dot(list(encoded_context, encoded_target), axes = 2)

output_entropy <- layer_dot(list(scalar_product,pos_neg_input), axes = 1)%>%
	layer_activation(activation = "sigmoid") %>%
	layer_reshape(target = c(1))
	
#creo e compilo il modello usando la binary crossentropy come loss function
model <- keras_model(list(context_input, target_input, pos_neg_input), output_entropy)

optimizer <- optimizer_adam(lr = 0.0001)

model %>% compile(
optimizer = optimizer,
loss = "binary_crossentropy"
)

n_rec <- 60
len <- 2*(maxlen*window_size - window_size*(window_size+1)/2) * (negative_samples+1)

for (l in 40:100){
	context_words <- numeric(len*n_rec)
	target_words <- numeric(len*n_rec)
	input_labels <- numeric(len*n_rec)

	#creo una matrice per tenere in memoria la classe dei documenti, a ogni classe associo un vettore della base canonica.
	#Suppongo che in input le classi dei documenti siano rappresentate da numeri interi e di sapere a priori quante sono le classi.
	document_class <- matrix(0,n_classes,1.1*len*n_rec)
	
	i <- 0
	for (n in 1:n_rec){
		document <- train_data[[n_rec*l + n]]
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
		}
	one_vector <- rep(1, i)

	#alleno il modello usando come output input_labels, che sarebbe l'array di zeri e uni creato da skiprams. Ho provato un numero di epoche diverse, ma dopo 5 tende a crescere la loss del training set
	model %>% fit(
	list(context_words, target_words, input_labels), one_vector,
	epochs = 1, batch_size = 768
	)
	#fine allenamento
	}
	
model %>% save_model_hdf5("skipgram_reuters_model.h5")