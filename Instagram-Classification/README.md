# Applied Data Analysis project

## Instagram Classification

Details for this project part can be found in [Instagram-Classification](https://nbviewer.jupyter.org/github/korcek-juraj/epfl-ada16-project/blob/master/Instagram-Classification/Instagram-Classification.ipynb) notebook. It will also lead you  step by step to reproduce the results. 

To run the code successfully you will need python3 (the code was tested on python3.5) along with [NumPy](https://pypi.python.org/pypi/numpy) and [TensorFlow](https://www.tensorflow.org/) libraries specified in requirements.txt.

If you do not have python3, you can download and install it from [python.org](https://www.python.org/downloads/). 

If you are running the code on your laptop you can install library requirements with following command. 
```
$ pip install requirements.txt
```
For Amazon AWS EC2 servers we suggest using [Machine Learning AMI](https://aws.amazon.com/marketplace/pp/B01M0AXXQB) (Amazon Machine Image) which has all the necessary machine learning libraries pre-installed.

If you want to run script from shell instead of the notebook just replace `%run ./script.py` with `python3 script.py`.  
For example:
```
%run ./create_url_dict.py -f months/september-october_urls.txt
```
will become:
```
python3 create_url_dict.py -f months/september-october_urls.txt
```

