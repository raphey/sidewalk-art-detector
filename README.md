# sidewalk-art-detector

AP wants to be able to find untagged photographs of his art. This is the ML component, aiming to pick out
photos of sidewalk art.

## Benchmarks

- ~model trains & distinguishes sidewalk art from other images with better than random performance~
- ~trained model wrapped in object that can accept an image URL as input and return likelihood~
- serve model such that API request with image URL returns 0-1 score for sidewalk art likelihood

## TODO list

- ~boilerplate poetry project~
- ~adapt [TF transfer learning collab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/transfer_learning.ipynb#scrollTo=ro4oYaEmxe4r)~
- ~check performance, share with AP~
- ~save/load model~
- ~create image-preprocessing function to match current TF preprocessing~
- ~make Predictor object that can be given a url and returns prediction~
- clean up train.py
- find best serving option (AWS SM? something cheaper?)
- make Dockerfile for serving
