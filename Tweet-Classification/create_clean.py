import pandas as pd
import re
import numpy as np
import enchant
import itertools
from urllib.parse import urlparse
from nltk.tokenize.casual import reduce_lengthening
import tldextract


def cleaning(month):
    '''
    Function that given a month will clean it
    Example :
    Input : the month
    Output : dataframe that contains the cleaned tweets with
    'source_location', 'source_spam_probability', the cleaned tweets, and the language of the tweet as columns
    '''
    i=0
    prob=[]
    sentiment=[]
    tweet=[]
    location=[]
    lang=[]
    for j in range(30):
        try:
            if j<9:
                file=pd.read_json(month[0]+"/harvest3r_twitter_data_0"+str(j+1)+"-"+month[1]+"_0.json")
            else:
                file=pd.read_json(month[0]+"/harvest3r_twitter_data_"+str(j+1)+"-"+month[1]+"_0.json")
            for line in file['_source']:
                if line['lang']!='U' and 'sentiment' in line.keys():
                    tweet.append(clean(line['main']))
                    sentiment.append(line['sentiment'])
                    #time.append(line['date_found'].split("T")[1][:-1])
                    if 'source_spam_probability' in line.keys():
                        prob.append(line['source_spam_probability'])
                    else:
                        prob.append("NaN")
                    #lang.append(line['lang'])
                    if 'source_location' in line.keys():
                        location.append(line['source_location'])
                    else:
                        location.append("NaN")

                    i+=1
        except:
            continue

    print(i)
    print(len(tweet))
    print(len(sentiment))
    print(len(location))
    
    #Putting everything in a dataframe
    df1=pd.DataFrame([])
    df1['cleaned-tweets']=tweet
    df1['sentiment']=sentiment
    df1['proba-spam']=prob
    df1['location']=location
    #df1['langue']=lang

    return df1


#Reading the 3-dictionnaries to de-noise the Tweets

dico = {}
dico1 = open('dicos/dico1.txt', 'rb')
for word in dico1:
    word = word.decode('utf8')
    word = word.split()
    dico[word[1]] = word[3]
dico1.close()
dico2 = open('dicos/dico2.txt', 'rb')
for word in dico2:
    word = word.decode('utf8')
    word = word.split()
    dico[word[0]] = word[1]
dico2.close()
dico3 = open('dicos/dico2.txt', 'rb')
for word in dico3:
    word = word.decode('utf8')
    word = word.split()
    dico[word[0]] = word[1]
dico3.close()



'''
Replace twitter pics urls by <pic>.
Example :
Input : "photo of me with my brother pic.twitter.com/S1FAM7BIi"
Output : "photo of me with my brother <pic>"
'''
def replacePics(tweet):
    tweet=re.sub(r"pic\.\S+"," <pic> ",tweet)
    return tweet

'''
Replace twitter mentions by <mention>.
Example :
Input : "Good morning @hamza ! I am with @Yassine."
Output : "Good morning <mention> ! I am with <mention>."
'''
def replaceMentions(tweet):
    tweet=re.sub(r"@\S+"," <mention> ",tweet)
    return tweet

def remove_repetitions(tweet):
    """
    Functions that remove noisy character repetition like for instance :
    llloooooooovvvvvve ====> love
    This function reduce the number of character repetition to 2 and checks if the word belong the english
    vocabulary by use of pyEnchant and reduce the number of character repetition to 1 otherwise
    Arguments: tweet (the tweet)
    """
    tweet=tweet.split()
    for i in range(len(tweet)):
        tweet[i]=''.join(''.join(s)[:3] for _, s in itertools.groupby(tweet[i])).replace('#', '')
        #if len(tweet[i])>0:
            #if not d.check(tweet[i]):
                #tweet[i] = ''.join(''.join(s)[:1] for _, s in itertools.groupby(tweet[i])).replace('#', '')
    tweet=' '.join(tweet)
    return tweet

def correct_spell(tweet):
    """
    Function that uses the three dictionaries that we described above and replace noisy words
    Arguments: tweet (the tweet)
    """
    tweet = tweet.split()
    for i in range(len(tweet)):
        if tweet[i] in dico.keys():
            tweet[i] = dico[tweet[i]]
    tweet = ' '.join(tweet)
    return tweet

def arr(tweet):
    """
    Function that cleans the tweet using the functions above and some regular expressions
    to reduce the noise
    Arguments: tweet (the tweet)
    """
    #Separates the contractions and the punctuation
    tweet = re.sub(r"\'s", " \'s", tweet)
    tweet = re.sub(r"\'ve", " \'ve", tweet)
    tweet = re.sub(r"n\'t", " n\'t", tweet)
    tweet = re.sub(r"\'re", " \'re", tweet)
    tweet = re.sub(r"\'d", " \'d", tweet)
    tweet = re.sub(r"\'ll", " \'ll", tweet)
    tweet = re.sub(r",", " , ", tweet)
    tweet = re.sub(r"!", " ! ", tweet)
    tweet = re.sub(r"\(", " \( ", tweet)
    tweet = re.sub(r"\)", " \) ", tweet)
    tweet = re.sub(r"\?", " \? ", tweet)
    tweet = re.sub(r"\s{2,}", " ", tweet)
    #tweet = correct_spell(tweet)
    return tweet.strip().lower()

'''
Replace url by its domain name.
Example :
Input : "I am using https://www.google.ch and https://fr.yahoo.com/"
Output : "I am using google and yahoo"
'''
def replaceURLsbyDomains(tweet):
    urls=re.findall(r"http\S+",tweet)
    for url in urls:
        tweet=tweet.replace(url,' <'+tldextract.extract(url).domain+'> ')
    return tweet


def clean(tweet):
    '''
    Function that cleans the tweets
    Input : tweet
    Output : cleaned tweet
    This function is called by the function cleaning above
    '''
    tweet=tweet.replace(";"," ")
    tweet=remove_repetitions(tweet)
    tweet=arr(tweet)
    tweet=replacePics(tweet)
    tweet=replaceMentions(tweet)
    tweet=replaceURLsbyDomains(tweet)
    #tweet=correct_spell(tweet)
    liste=[word for word in tweet.split() if len(word)>1]
    tweet=' '.join(liste)
    return tweet.strip().lower()


seasons=[[("february","02"),("january","01")],[("march","03"),("april","04"),("may","05")],[("july","07"),("june","06"),("august","08")],[("october","10"),
("september","09")]]
seasonsfiles=[]
# Running the cleaning over the files and creating seasonal cleaned files
#1.csv, 2.csv, 3.csv and 4.csv
for j in range(len(seasons)):
    for i in range(len(seasons[j])):
        if i==0:
            seasonsfiles.append(cleaning(seasons[j][i]))
        else:
            seasonsfiles[j]=pd.concat([seasonsfiles[j],cleaning(seasons[j][i])]).reset_index(drop=True)
    seasonsfiles[j].to_csv("cleaned-data/"+str(j+1)+".csv",sep=';')
