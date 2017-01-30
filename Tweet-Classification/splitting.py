import pandas as pd


cols = ["geonameid","name","ascii","alternate","latitude","longitude","fclass","fcode","ccode","cc2",
           "ad1","ad2","ad3","ad4","pop","elev","dem","timeZ","date"]
df_ = pd.read_csv("CH.txt",sep="\t", header=None, names=cols)
del df_['geonameid'], df_['ascii'], df_['fclass'], df_['fcode'], df_['ccode'], df_['cc2'], df_['ad1']
del df_['ad2'], df_['ad3'], df_['ad4'], df_['elev'], df_['dem'], df_['timeZ'], df_['date']
df = df_.loc[df_['pop']>10000]
del df['pop']
df = df.dropna()
df = df.reset_index()
del df['index']
cities = {}
for i in range(df.shape[0]):
    if df['name'][i]!='Switzerland':
        subcities = df['alternate'][i].split(',')
        for sub in subcities:
            cities[sub]=(df['latitude'][i],df['longitude'][i])


files=["s1-cleaned/1","s2-cleaned/2","s3-cleaned/3","s4-cleaned/4"]

for file in files:
    file1=pd.read_csv("cleaned-data/"+file+".csv",sep=';')
    del file1['Unnamed: 0']
    train_labels = []
    train_tweets = []
    test_tweets = []
    test_labels = []
    test_lats = []
    test_longs = []
    test_probas = []
    neutral_tweets = []
    neutral_labels = []
    for i in range(file1.shape[0]):
        if file1['location'][i] in cities:
            if file1['sentiment'][i]=='NEUTRAL':
                neutral_tweets.append(file1['cleaned-tweets'][i])
                neutral_labels.append('test')
            else:
                test_tweets.append(file1['cleaned-tweets'][i])
                if file1['sentiment'][i]=='POSITIVE':
                    test_labels.append(0)
                else:
                    test_labels.append(1)
                test_lats.append(cities[file1['location'][i]][0])
                test_longs.append(cities[file1['location'][i]][1])
                test_probas.append(file1['proba-spam'][i])
        else:
            if file1['sentiment'][i]=='NEUTRAL':
                neutral_tweets.append(file1['cleaned-tweets'][i])
                neutral_labels.append('train')
            else:
                train_tweets.append(file1['cleaned-tweets'][i])
                if file1['sentiment'][i]=='POSITIVE':
                    train_labels.append(0)
                else:
                    train_labels.append(1)
    df_train=pd.DataFrame([])
    df_test=pd.DataFrame([])
    df_train['label']=train_labels
    df_train['tweet']=train_tweets
    df_test['label']=test_labels
    df_test['tweet']=test_tweets
    df_test['latitude']=test_lats
    df_test['longitude']=test_longs
    df_test['proba']=test_probas
    df_n = pd.DataFrame([])
    df_n['tweet']=neutral_tweets
    df_n['label']=neutral_labels
    df_train.to_csv("cleaned-data/"+file+"train.csv",sep=';')
    df_test.to_csv("cleaned-data/"+file+"test.csv",sep=';')
    df_n.to_csv("cleaned-data/"+file+"neutral.csv",sep=';')
