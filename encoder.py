#Import neccessary modules
import numpy as np
import string
from keras.datasets.imdb import load_data, get_word_index
from keras import Input, Sequential, Model
from keras.layers import Layer, Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from keras.preprocessing.sequence import pad_sequences

#Only count the 15000 most common words, and get the data along with the word2int vocabulary
NUM_WORDS = 15000
(X_train, y_train), (X_test, y_test) = load_data(num_words=NUM_WORDS)
word2int = get_word_index()

MAX_LEN = 200
#Pad the X data such that each vector is transformed to MAX_LEN(200) dimensions
X_train = pad_sequences(X_train, maxlen=MAX_LEN)
X_test = pad_sequences(X_test, maxlen=MAX_LEN)
#Change the y data to numpy arrays and reshape them for the neural network
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)

#Manually create a Positional Embedding Layer as shown by the paper, which brings out the notion of order to the transformer model
#All words of the input sequence are fed into the model at once, so it does not understand ordering of words and needs Positional Embeddings
class PositionalEmbedding(Layer):
    def __init__(self, input_dim, input_length):
        super(PositionalEmbedding, self).__init__()

        #Make an array with input_length and the dimensions of the model, and use the mathematical operation in the transformer paper. 
        #The array being made up of zeros does not change anything, as the positional embeddings are purely based on their index, not their value
        self.pos_embeddings = np.zeros((input_length, input_dim))
        for pos in range(input_length):
            for i in range(0, input_dim, 2):
                self.pos_embeddings[pos, i] = np.sin(pos / (10000 ** ((2*i)/input_dim)))
                self.pos_embeddings[pos, i+1] = np.cos(pos / (10000 ** ((2*(i+1))/input_dim)))

    #Calling the Positional Embedding layer will just return the pos_embeddings numpy array
    def call(self, x):
        return self.pos_embeddings

#In Transformers, the normal Embedding layer output of the input sequence is added with the output of the Positional Embedding output
class TransformerEmbedding(Layer):
    def __init__(self, input_dim, output_dim, input_length, weights=None):
        super(TransformerEmbedding, self).__init__()

        self.total_embeddings = np.zeros((input_length, output_dim))
        
        #Get the Positional Embeddings of an array with dimensions input_length by output_dim
        self.pos_embeddings = PositionalEmbedding(input_dim=output_dim, input_length=input_length)(self.total_embeddings)
        #Make an Embedding layer with either no pretrained vectors or pretrained vectors
        if weights != None:
            self.embeddings = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length, weights=weights, trainable=False)
        else:
            self.embeddings = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length)
        
    #When calling this layer, it will get the regular Embedding vectors in the input sequence, add it with the positional embeddings, and return it
    def call(self, x):
        x = self.embeddings(x)
        self.total_embeddings = x + self.pos_embeddings
        return self.total_embeddings

#In Transformers, there needs to be a Feed Forward neural network to processing the inputs from the Attention layers to something processible by a neural network
class FeedForward(Layer):
    def __init__(self, units, input_dim, dropout_rate):
        super(FeedForward, self).__init__()

        #The Feed Forward net will consist of one Linear Layer with any amount of units, a dropout, and then a final linear layer with the embedding dimensions
        #The final linear layer has to have the embedding dimensions so the outputs can be combined with the attention-normalization part of the Encoder
        self.ff = Sequential([
            Dense(units, activation='relu'),
            Dropout(dropout_rate),
            Dense(input_dim)
        ])

    #When called, this layer will process the inputs and return them
    def call(self, x):
        return self.ff(x)

#The Encoder Block is the neural net block which processing the embeddings and outputs the resulting vectors, which can be averaged and used for the output layer
class EncoderBlock(Layer):
    def __init__(self, ff_units, embedding_dim, num_words, n_heads, dropout_rate=0):
        super(EncoderBlock, self).__init__()
        
        self.ff = FeedForward(units=ff_units, input_dim=embedding_dim, dropout_rate=dropout_rate)
        #Multi Head Attention in transformers gives the transformer an understanding of how each word is dependent on the rest of the words in the sentence
        #However, attention layers for each word often output their respective word as unreasonably important, not giving much information on the other words
        #So, the outputs of the attention layer are repeated n_heads amount of times, and averaged out  
        self.attention = MultiHeadAttention(num_heads=n_heads, key_dim=num_words)
        self.norm1, self.norm2 = LayerNormalization(), LayerNormalization()
        self.dropout1, self.dropout2 = Dropout(dropout_rate), Dropout(dropout_rate)

    def call(self, x):
        #The first half of the call method consists of attentionizing and normalizing the inputs, normalizing the inputs, and adding those two together
        x1 = self.attention(x, value=x)
        x1 = self.norm1(x1)
        x2 = self.norm1(x)
        x3 = x1 + x2
        #Add a dropout to the new outputs
        x3 = self.dropout1(x3)
        
        #The second half of the call method consists of Linearly processing and normalizing the inputs, normalizing the inputs, and adding those two together
        x4 = self.ff(x3)
        x4 = self.norm2(x4)
        x5 = self.norm2(x3)
        x6 = x4 + x5
        #Add a dropout to the new outputs
        x6 = self.dropout2(x6)

        #Return the resulting output vectors
        return x6

#The Embedding dimension used for this will be 36 dimensions, as ones of a higher value would be unimportant and costly to time
EMBEDDING_DIM = 36

#When manually making layers, you have to use keras functional API, so we will specify the input shape
encoder_input = Input(shape=(MAX_LEN,))
#Use the TransformerEmbedding layer to transform each unique token to a EMBEDDING_DIM(36) dimensional vector
encoder = TransformerEmbedding(input_dim=NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN)(encoder_input)
#In this neural network, we will use one encoder block with a 36 neuron feed forward net and a num_heads as 2 for simplistic reasons
#We could stack as many encoder blocks as we wanted on top of each other, but that would be time costly and we can get enough info from one
encoder = EncoderBlock(ff_units=36, embedding_dim=EMBEDDING_DIM, num_words=NUM_WORDS, n_heads=2)(encoder)
#In order to flatten the outputs from the encoder, we will use GlobalAveragePooling, so we can keep the main features from the encoder output
encoder = GlobalAveragePooling1D()(encoder)
#In transformers, there is a Linear Layer before the final layer, so we will use a 36 neuron Linear layer and a 10% Dropout to reduce overfitting
encoder = Dense(36, activation='relu')(encoder)
encoder = Dropout(0.1)(encoder)
#Since classifying movie reviews is a binary classification problem, we can use a one neuron sigmoid activation as our final layer
encoder_output = Dense(1, activation='sigmoid')(encoder)

#Create the model with the inputs and outputs, and using the Adam optimizer we can compile it and get ready to train it
model = Model(inputs=encoder_input, outputs=encoder_output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#Fit the dataset on the training data, and test on the testing data(got an 88.24% validation accuracy)
model.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_test, y_test))
