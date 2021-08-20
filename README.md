# Multiclass-Text-Classification-with-DistilBERT-on-COVID-19-Tweets  
### Corona Virus Tagged Data

### Task  
Perform Text Classification on the tweets pulled from Twitter.

### Link to data
https://www.kaggle.com/datatattle/covid-19-nlp-text-classification

#### Labels (5 Categories)
- Extremely Positive
- Positive
- Neutral
- Negative
- Extremely Negative

#### Labels (3 Categories)
- Positive
- Neutral
- Negative

### Description of work done
In this notebook, I implement a Naive Bayes model (for a baseline score to beat) and a Deep Neural Network (DNN) using DistilBERT, LSTM, and Dense layers for both the 5 and 3 category labels. I had to pre-process the entire dataset by Label Encoding the sentiment, splitting data from the training set to be validation data, removing URL's in the tweets and dropping duplicate entries. For the Naive Bayes model, I had to further pre-process the data by removing all punctuation (Except for #'s), lowercase all the tweets, remove stopwords, and stemmed words. For the DNN, I had to pre-process all the training examples into BERT's proper input format before training by tokenizing each entry in the training, validation, and testing data along with getting the input masks. 

During the DNN training process, i implemented ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau to get the greatest results as possible. I visualized the training and validation accuracy and loss and loaded the weights of the epoch with the highest validation accuracy and made the predictions on the testing data with them. I then visualized the results using a heatmap and printed out the classification reports. 

Finally, I examined which predictions were incorrect on the DNN model with 3 categories. 
