# Identifying Spam Messages using NLP and Deep Learning Methods

<br>
<p align="center" width="100%">
<kbd><img src="images/email.png" width="300"  /></kbd>
</p>
<br>

# Context

This short project was completed over the course of a week in my spare time with the intetion of building further familiarity with some NLP methods and the Keras/TensorFlow library. To keep things simple I have lifted a pre-made dataset from [kaggle.com](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification).

# Spam Statistics

Approximately 22.5 billion legitimate emails are sent over the internet each day and this is only 15% of the total number of emails! Nearly 85% of allemails are spam, meaning **over 122 billion spam emails are sent daily**. 

Of these 122 billion spam emails, the [top three categories](https://dataprot.net/statistics/spam-statistics/) are:

1. 36% of spam emails are advertising 
2. 32% of spam emails contain adult content
3. 27% of spam emails are related to financial matters

Spam emails are sometimes dangerous, often costly and always annoying. Fortunately ML techniques can be used to identify and intercept spam before they reach our inboxes.

# Objectives

The primary objective is to build a deep learning model which can identify messages as spam or 'ham', i.e. not spam. Model performance should be evaluated in terms of accuracy, precision, recall and f1-score for the spam category. 

Although possible, for this problem we have no assumed underlying business case such as *maximise recall of spam emails*.

The secondary objective is simply to learn about new aspects of NLP and TensorFlow.  

# Input Data

To keep things simple I have lifted a pre-made dataset from [kaggle.com](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification), consisting of 5157 unique text messages. 12% of the messages are labelles as spam, leaving the remaining 88% as 'ham'. Although these are specifically text messages rather than emails the content is similar enough for the purposes of this project. 
<br>
<br>

| | Category  | Message |
|-| ------------- | ------------- |
|0| ham  | Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...  |
|1| ham  | Ok lar... Joking wif u oni...  |
|2| spam  | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's  |
|3| ham  | U dun say so early hor... U c already then say...  |
|4| ham  | Nah I don't think he goes to usf, he lives around here though |
<p align="center"><i><sub><b>Figure 1:</b> Top five rows of the input data.</sub></i></p>
<br>

# Data Cleaning and EDA

Since this is a two column dataset with `Category` and `Message` columns there is really very little data cleaning to do. Removing duplicates in the `Message` column identified 415 duplicated messages which were dropped to leave 5157 unique messages. Punctuation and text case will be standardised later as part of the NLP workflow so for now the cleaning is done. 

<br>
<p align="center" width="100%">
<kbd><img src="images/eda.png" width="900"  /></kbd>
</p>
<p align="center"><i><sub><b>Figure 2:</b> Left: Distribution plot of message wordcount by spam/ham label. Right: Pie chart illustration of dataset composition by spam/ham label.</sub></i></p>
<br>

The histogram in figure 2 above shows that the median word count for a spam message is around 25 words. Shorter messages around 12 words long are predominantly ham whilst the longest messages are also commonly ham. The distibution plots are nice, but we can also look at some figures to describe the data. 

<br>
<p align="center" width="100%">
<kbd><img src="images/stats.png" width="200"  /></kbd>
</p>
<p align="center"><i><sub><b>Figure 3:</b> Descriptive metrics by spam/ham classification.</sub></i></p>
<br>

The values in figure 3 align with what is observed in figure 2. Finally let's have a look at some word clouds for the different categories.

<br>
<p align="center" width="100%">
<kbd><img src="images/cloud2.png" width="900"  /></kbd>
</p>
<p align="center"><i><sub><b>Figure 4:</b> Word clouds for spam messages and ham messages. The largest words are the most frequntly present.</sub></i></p>
<br>

I used the `TfidfVectorizer` class from Scikit-learn to generate these word clouds. This method won't be used later in the modelling, it was just a bit of fun to investigate what sort of words are being used in the spam/ham. There's a sense of urgency in the spam cloud, with more capital letters and more prizes to be won. 

That concludes this brief interrogation of a very simple dataset. Although I have used word count to describe differences between the spam and the ham, this attribute will not be included in the spam prediction model. Instead I will look to use some NLP methods to generate predictors. 

# NLP and Modelling

Now we get to the learning part for me. My previous NLP experience has been captured [here](https://github.com/rgdavies92/salaries-in-data) in a General Assembly project concerning job salaries in data related roles. The workflow consisted of stemming words, removing stop-words, count-vectorising over n-grams and selecting the top n predictors for input to logistic regression. This worked quite well at the time but I'm going to try a different approach here.

The workflow to be implemented in this spam project consists of the following:
* Tokenize words in the messages to obtain integer vectors where each integer is the index of a token in a dictionary.
* Pad the sequences to ensure all integer vectors have a constant length of 100 integers.
* Use a pre-trained GloVe word embedding dataset in 100 dimensional space to map the tokenized words to the corresponding embedded vector. 
* Enter the GloVe embedded messages as predictors into a Bi-directional Long Short-Term Memory Recurrent Neural Network (BiLSTM RNN 🤯) model.

That's the high level overview of the process. I'll now try to break down each of those steps a little further, including my favourite references.

### Tokenize

The Tokenizer class from TensorFlow has allowed each message to be vectorised by turning it into a sequence of integers where each integer is the index of a token in a dictionary. The length of each tokenized message was equal to the number of words in the message. The default behaviour of Tokenizer is to split on white space and filter all punctuation, tabs and line breaks besides the ' character `(!"#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n)`, so I didn't have to manually standardise punctuation. Tokenizer also converts all words to lower-case as default, saving me a step in preprocessing. 

### Pad

Messages vary in length and so the tokenized outputs also very in length. This must be standardised before word embedding and can easily be done using the pad_sequences module again from TensorFlow. I have opted to cut all messages to 50 words/tokens in length based on the word count histogram in figure 2 above. Where messages are longer than 50 words, the first 50 words are retained, dropping the 51st onwards. My rationale is that the purpose of any message is usually at the start rather than the end. 

After sifting through documentation detailing what the tokenizer was doing and how it interacted with pad_sequences I found a nice and basic article on [kdnuggets](https://www.kdnuggets.com/2020/03/tensorflow-keras-tokenization-text-data-prep.html) which explains the motions with simple examples. I wish I found this at the start. The messages are now ready to be embedded.

### Prepare Word Embedding

Word embedding is a method by which we can represent words as dense vectors. The traditional bag-of-words models consisting of large sparse vectors can quickly grow to unmanagable dimensions, with very high feature redundancy. There are a number of methods by which we can reduce the dimensionality of these NLP problems and I'm going to use word embedding here.

I use a pre-trained GloVe (Global Vectors) word embedding for this problem - the vector represtation for each word has already been learned on a massive corpus of 6B tokens which is more than I could do! More can be read aboute GloVe [here](https://nlp.stanford.edu/projects/glove/). 

In this project I have opted to map each word to a 100 term vector. The selling point for word embedding is that it maps similar words to similar vectors. Based on the context in which these words are presented GloVe is able to map words to the 100 dimensional vector space in a way which retains information about their usage and relationships. I think of this as adding texture to the predictor dataset where a bag of words model would have very little texture. There are some cool examples of word relationships on the [GloVe site](https://nlp.stanford.edu/projects/glove/). 

[Helpful source of information.](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/) 

I note that at this stage I have only prepared the word embedding. It is actually implemented as the first stage of the model

### Train-Test Split

The messages are split to a train size of 75% and a test size of 25% for model evaluation. 

### Model

A BiLSTM RNN model was defined using the Sequential class from TensorFlow. Layers are stacked into the model in the order Embedding; 128 BiLSTM units; 0.3 Dropout; Dense Output with sigmoid activation function. It might be heavy reading, but a high level of detail has been included with comments in the  model definition code block below.

```Python
def get_bidirectional_model(tokenizer, lstm_units):
    """
    Constructs the model,
    Embedding vectors => Bi-LSTM => 1 fully-connected output neuron with sigmoid activation
    """
   
   # Get the GloVe embedding vectors
    embedding_matrix = get_embedding_vectors(tokenizer) 
                       # This is a function I've defined earlier to get the (9005,100) embedding matrix
    
    # Define the model sequentially https://machinelearningmastery.com/keras-functional-api-deep-learning
    model = Sequential()
    
    # First, embed the words using loaded GloVe
    model.add(Embedding(input_dim=len(tokenizer.word_index)+1, # In this case, we have 9005 tokens in the index
                        output_dim=100, # Each output word is embedded as a vector of length 100
                        weights=[embedding_matrix],
                        trainable=True, # Allow updating if the word embeddings can be improved. 
                        mask_zero=True, # I'm not intertested in the zero values which were padded
                        input_length=50)) # Messages were limited to 50 words based on figure 2
    
    # Add bidirectional long short-term memory units
    model.add(Bidirectional(LSTM(lstm_units, recurrent_dropout=0.2)))
    
    # Add dropout to combat overfitting.
    model.add(Dropout(0.2))
    
    # Add output dense layer with sigmoid for 1/0 classification
    model.add(Dense(1, activation="sigmoid"))
    
    # Compile as rmsprop optimizer and pre defined metrics
    model.compile(optimizer="rmsprop", # A first order, gradient-based optimiser which uses an adaptive learning rate
                  loss="binary_crossentropy", # Equivalent to the log-loss function
                  metrics=[METRICS])

    model.summary()
    return model
    
# Build the bidirectional-model with 128 LSTM units
bimodel = get_bidirectional_model(tokenizer=tokenizer, lstm_units=128)
```

Now some of the links that I found usefull in defining this:
* [What is the vanishing gradient problema nd what is a LSTM](https://towardsdatascience.com/introduction-to-lstm-units-while-playing-jazz-fa0175b59012#)
* [Understanding why I might want to used a BiLSTM rather than an LSTM](https://arxiv.org/pdf/1801.02143.pdf)
* [Implementing dropout](https://towardsdatascience.com/understanding-and-implementing-dropout-in-tensorflow-and-keras-a8a3a02c1bfa)
* [Recurrent dropout as opposed to standard dropout](https://arxiv.org/pdf/1512.05287.pdf)
* [The binary crossentropy loss function](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)
* [The rmsprop optimiser](https://medium.com/analytics-vidhya/a-complete-guide-to-adam-and-rmsprop-optimizer-75f4502d83be)

The only other point I'd like to add to this model definition is why I was compelled to choose a BiLSTM RNN in the first place. There are a few reasons: 
* A RNN has a recurrent connection on the hidden state which allows sequential information to be captured in the input data. This makes it particularly useful for text problems where the sequence is paramount. 
* With training the model through back-propagation RNNs suffer from what is known as the vanishing gradient problem. This means that the RNN would struggle to learn long-range dependencies from the early layers and is often described as a short-term memory problem. To combat this I tried using specialised units in the hidden layers: Long Short-Term Memory units or LSTMs. These LSTMs are able to learn long-range dependencies through a series of gated tensor operations which dictate what information to add or remove from the hidden state at each unit.
* Finally, I moved from LSMTs to BiLSTMs because in the spam problem within which I'm working the preceding text is as important as the succeeding text. Word context is a bi-directional feature. Interestingly I don't think this delivered much model uplift, but the theory is strong so I've retained the BiLSTM units.  

<br>
<p align="center" width="100%">
<kbd><img src="images/units.png" width="700"  /></kbd>
</p>
<p align="center"><i><sub><b>Figure 5:</b> Diagrams of the bi-directional implementation of LSTM units and single LSTM unit architecture. Within the LSTM unit, the pink cicrcles are arithmetic operators and the coloured rectangles are the gates in LSTM. Sigma denotes the sigmoid function and tanh denotes the hyperbolic tangent function. Images from Cui, Z., Ke, R., Pu, Z. and Wang, Y., 2018. Deep bidirectional and unidirectional LSTM recurrent neural network for network-wide traffic speed prediction. arXiv preprint arXiv:1801.02143. https://arxiv.org/pdf/1801.02143.pdf</sub></i></p>
<br>

# Results

# Closing thoughts

I'm a little concerned about how the GloVe embedding will handle 'text speak' where characters were once limited. It might be that this model actually performs better on email spam than text spam. Maybe it's not true that all spam is equal.

If the BiLSTM model pays heeds to preceding and succeeding context, then might it be important to retain some of the punctuation through tokenization? Punctuation is instrumental in providing context and so might be beneficial with this BiLSTM RNN model. I'll be sure to test this if I'm ever working on something like this in a production scenario.

Back on my spam problem, the first step was to use the `Tokenizer` class and `pad_sequences` module from TensorFlow and Keras. The Tokenizer class has allowed each message to be vectorised by turning it into a sequence of integers where each integer is the index of a token in a dictionary. The default behaviour of Tokenizer is to filter all punctuation besides the ' character and convert all text to lower-case.
