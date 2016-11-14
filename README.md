# Applied Data Analysis project

Aim : Doing a happy map for a city in Switzerland like GoodCityLife project (NOKIA)  
Idea : The idea would be to build a happy map minimizing both the inverse of happiness and the travel time  

Work Plan :

We suppose that we will have at our disposal :

- large twitter data (geolocated)

- instagram images (geolocated)

1/ Analysis :

- study our dataset

- perform a sentiment analysis using convolutional / recurrent deep neural networks for Twitter data (also NLP methods like word embedding)

- perform sentiment analysis using convolutional deep networks on instagram images: the advantage of convolutional neural network is that we don’t need to preprocess the image and extract relevant features we can feed the whole image to the model and optimize the parameters

- for the analysis we will be using Keras

2/ Viz : 

- optimization (Algorithm)

- build a map
         
3/ Build a website that outputs the path found by the algorithm depending on user’s destination/provenance
