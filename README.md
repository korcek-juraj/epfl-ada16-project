Aim : Doing a happy map for a city in Switzerland like GoodCityLife project (NOKIA).
Idea : The idea would be to build a happy map minimizing both the inverse of happiness and the travel time.
Work Plan :

We suppose that we will have at our disposal :

- Large twitter data (geolocated)

- instagram images (geolocated)

1/ Analysis :

- study our dataset

-  perform a sentiment analysis using Convolutional/recurrent deep neural networks for Twitter data(also NLP methods like word embedding).

- Perform sentiment analysis using convolutional deep networks on instagram images : The advantage of convolutional neural network is that we don’t need to preprocess the image and extract relevant features we can feed the whole image to the model and optimize the parameters- For the analysis we will be using Keras.

2/ Viz : - Optimization (Algorithm)

         - Build a map.
         
3/ Build a website that outputs the path found by the algorithm depending on the user’s destination/provenance
