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

Since this is a two column dataset, with `Category` and `Message` columns there is really very little data cleaning to do. Removing duplicates in the `Message` column identified 415 duplicated messages which were dropped to leave 5157 unique messages. Punctuation and text case will be standardised later as part of the NLP workflow so for now the cleaning is done. 

<br>
<p align="center" width="100%">
<kbd><img src="images/eda.png" width="600"  /></kbd>
</p>
<p align="center"><i><sub><b>Figure 2:</b> Left:Distribution plot of email wordcount by spam/ham label. Right: Pie chart illustration of dataset composition by spam/ham label.</sub></i></p>
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
<kbd><img src="images/clouds.png" width="600"  /></kbd>
</p>
<p align="center"><i><sub><b>Figure 4:</b> Word clouds for all messages, spam messages and ham messages.</sub></i></p>
<br>

I used a the `TfidfVectorizer` class from Scikit-learn to generate these word clouds. This method won't be used later in the modelling, it was just a bit of fun to investigate what sort of words are being used in the spam/ham. There's a sense of urgency in the spam cloud, with more capital letters and more prizes to be won. 

That concludes this brief interrogation of a very simple dataset. Although I have used word count to describe differences between the spam and the ham, this attribute will not be included in the spam prediction model. Instead I will look to use some NLP methods to generate predictors. 

# NLP and Modelling

Now we get to the learning part for me. My previous NLP experience has been captured [here](https://github.com/rgdavies92/salaries-in-data) in a General Assembly project concerning job salaries in data related roles. The workflow consisted of stemming words, removing stop-words, count-vectorising over n-grams, selecting the top n predictors for input to logistic regression. This worked quite well at the time but I'm going to try a different approach here.

The workflow to be implemented in this spam project consists of the following:
* Tokenize words in the messages to obtain integer vectors where each integer is the index of a token in a dictionary.
* Pad the sequences to ensure all integer vectors have a constant length of 100 integers.
* Use a pre-trained GloVe word embedding dataset in 100 dimensional space to map the tokenized words to the corresponding embedded vector. 
* Enter the GloVe embedded messages as predictors into a Bi-directional Long Short-Term Memory Recurrent Neural Network (BiLSTM RNN 🤯) model

That's the high level overview of the process. I'll now try to break down each of those steps a little further, including my favourite references.

## Tokenize

## Pad

## Embed

## Model

# Results

# Closing thoughts

I'm a little concerned about how the GloVe embedding will handle 'text speak' where characters were once limited. It might be that this model actually performs better on email spam than text spam. Maybe it's not true that all spam is equal.

Back on my spam problem, the first step was to use the `Tokenizer` class and `pad_sequences` module from TensorFlow and Keras. The Tokenizer class has allowed each message to be vectorised by turning it into a sequence of integers where each integer is the index of a token in a dictionary. The default behaviour of Tokenizer is to filter all punctuation besides the ' character and convert all text to lower-case.
