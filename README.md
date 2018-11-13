# tesigalileiana

This code is part of the thesis of Gianluca Maguolo for the Scuola Galileiana di Studi Superiori
For any problem with the code, contact Gianluca Maguolo at gianluca.maguolo@phd.unipd.it

***********************************************************************************
glove_model.m is an implementation of glove. The hyperparamters can be chosen at the beginning. The script returns the most similar words to given "center" words chosen in the script.

***********************************************************************************
skip_gram_model.m is an implementation of Skip-Gram. The script trains the model and returns the most similar words to given "center" words. It saves the model in the same directory.

***********************************************************************************
reuters_bag_of_embeddings.m is an implementation of bag of embeddings usful for document classification. It initializes the model with the weights of Skip-Gram, so skip_gram_model.m must have run before. The script trains the model and returns the most similar words to given "center" words.

***********************************************************************************
document_classification.m uses the weights coming from reuters_bag_of_embeddings.m to perform document classification. The script makes an unbalanced, 8 class classification.
