import _pickle as cPickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling1D
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from featuring import dumpFeatures
import pandas as pd
from sklearn.metrics import classification_report
import csv



"""
Dumps the features
"""
#dumpFeatures( n_gram=False, pretrained=True, nb_words=None, namefile='pretrained_features3.dat')

"""
Load the features from the pickled ones
"""

[X_train, y,ytest, X_test, max_features, W] = cPickle.load(open("pretrained_features3.dat", "rb"))

"""
Set the seed for model
"""
np.random.seed(1)




"""
Using keras, we define the first model with one embedding layer and
one convolutional layer followed by one maxpooling layer and 2 Dense layers
with both reLu and sigmoid activation functions
Here for model 1 to 5 we used the glove200 pretrained embedding (200 stands for the dimension of the word vectors)
weights=[W] is the argument given to the embedding W is then the matrix built using glove
Also, for all models we used binary_crossentropy as a measure of the loss and
after testing some other optimizers like adadelta we chose to fit all our models with Adam optimizer
with default learning rate of 0.001
"""
model = Sequential()
model.add(Embedding(max_features+1, 200, input_length=X_train.shape[1], weights=[W]))
#model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
#model.add(MaxPooling1D(pool_length=2))
#model.add(Flatten())
#model.add(Dense(250, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=128, verbose=1, shuffle=True)



test_10 = model.predict_proba(X_test)
out = csv.writer(open("pred-prob3.csv","w"), delimiter=',',quoting=csv.QUOTE_ALL)
out.writerow(test_10)
test_20=model.predict_classes(X_test)
#for j in np.linspace(0.46,0.54 ,10):
#print("-----------------------------------------"+str(j)+"------------------------------------")
#test_20=[int(test_10[i]>j) for i in range(len(test_10))]
print(classification_report(test_20,ytest))
