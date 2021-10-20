# sidewalk-art-detector

AP wants to be able to find untagged photographs of his art. This is the ML component, aiming to pick out
photos of sidewalk art.

## Benchmarks

- model trains & distinguishes sidewalk art from other images with better than random performance
- serve model such that API request with image URL returns 0-1 score for sidewalk art likelihood

## TODO list

- ~boilerplate poetry project~
- adapt [TF transfer learning collab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/transfer_learning.ipynb#scrollTo=ro4oYaEmxe4r)
- check performance, share with AP
- if performance sufficient, refactor to use handmade image preprocessing function for training data & serving
- find best serving option (AWS SM? something cheaper?)
- make Dockerfile for serving
