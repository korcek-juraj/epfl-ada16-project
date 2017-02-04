# Sentiment Analysis on Twitter data 

For the first part of this project, we did sentiment analysis on twitter data. we had at our disposal daily geolocated tweets in Switzerland
from January to October. These tweets were labeled by "POSITIVE", "NEGATIVE", or "NEUTRAL". The end-goal of this project is to scrape the 
web daily and run on them the algorithms we found to build a daily sentiment map of Switzerland.

## Methods

- The first part before running any analysis was to clean the data, since Tweets are very noisy texts. You can see the pipeline for the cleaning
in the file `create_clean.py`. After that, we analyzed the data and we noticed the huge imbalance. In fact, doing a 3-class classification 
without considering the imbalance would be the same as running a dummy model that always predict the majority class (e.g. "NEUTRAL" class).
We decided to train the models only on a POSITIVE/NEGATIVE labeled tweets. Also, since the features might change from one season to another,
we decided to do seasonal models : 1/ January, February 2/ March, April, May 3/ June, July, August 3/ September, October


- In the folder `dicos` you can find the three dictionaries we found to help with de-noising the text.

- Then, we had to split our data into training/validation/test. With the file `splitting.py`, and since all tweets weren't geolocated, we
stored all the tweets that are geolocated into the test set since we only need the geolocation to build the map. We splitted the rest into 10%
for validation 90% for training. To get the latitude/longitude, we used GEONAMES. In fact, the source_location was given into several languages
and GEONAMES is a database that covers many places in switzerland in several languages.

- In the file `featuring.py`, you can find the two sets of features we used : 
- `Pre-trained embedding`: 1-grams with pretrained word embeddings (GLoVE)
- `1,2-gram`: 1,2-gram features

For both sets of features, we used word embeddings because of the large amount of data we had. And word embeddings with GLoVE have proven to be 
very effective. We pickled the resulting features. We also noticed that the Pre-trained embedding gave better results than using 1,2 grams.

- The file `models.py` contains the final models we used.

- With Keras, we used 2 models : convolutional neural net and fasttext. Actually, convolutional neural networks with pre-trained word embeddings
gave us very good results.

## Results

- Here are the results we obtained using convolutional neural networks on a 2-class classification for the first season (January and February).

| Class        | Precision           | Recall              | F1-score            |
|--------------|:-------------------:|:-------------------:|:-------------------:|
| Positive     | 0.95                | 0.96                | 0.96                |
| Negative     | 0.89                | 0.85                | 0.87                |

These are very good results, and after returning back to the data, we noticed that most of the tweets that contains a positive smiley are labeled
"POSITIVE" and the same for negative smileys. As for the neutral tweets they do not contain any smileys. That's why, we decided to remove all the
smileys and to run a word-based model and here are the results we obtained with the same model :

| Class        | Precision           | Recall              | F1-score            |
|--------------|:-------------------:|:-------------------:|:-------------------:|
| Positive     | 0.96                | 0.92                | 0.94                |
| Negative     | 0.73                | 0.84                | 0.78                |

Actually, we notice a little drop on the second class and this is due to the fact that the data without the "NEUTRAL" tweets is slightly imbalanced
toward "POSITIVE" labeled tweets and removing the strongest feature affected the predictions on this class. Still, the results good and very good
for the other seasons.

Finally, and since scraping the web would result in many "NEUTRAL" tweets, we decided to test our model on the "NEUTRAL" Tweets.
In the folder images, you can find the prediction probabilities on a 2-class classification on the "NEUTRAL" and "POS/NEG" Tweets. As you can
see, the classification probabilities are very high for "POS/NEG" Tweets and more uncertain on "NEUTRAL" Tweets. That's why we could get rid
of most of neutral Tweets with a high threshold.
## How to run the code

```
$ pip install requirements.txt
```

First put all the data in a folder in which you create the folder `cleaned-data` and create all the cleaned files :
```
$ python3 create_clean.py
```
This will create files with cleaned data by season.

Then after installing GEONAMES, do the splitting into training/test set by mean of : 

```
$ python3 splitting.py
```

In `featuring.py` you can specify in which season you want to train the models in the variable filetrain and filetest

Finally run the model with :

```
$ python3 models.py
```
This will pickle the features of the model and run it over the traning set and test it over the set. It will yield a classification report.
