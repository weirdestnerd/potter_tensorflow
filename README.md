# Potter

A model for predicting whether or not a spell in the Harry Potter book series was used in the seventh book.

## Inspiration

This project was inspired by Code2040's 2017 Fellow application. The datasets on spells was provided by [Code2040](https://fellows.code2040.org) (in two forms: spells & mentions) and modified by me to fit the training method used (Linear Regression).

## Dataset

The datasets were preprocessed in Ruby (I have reasons why. The code will be transformed to Python in the future.) The datasets has the following features:

* Classification [charm, curse, jinx, spell]: Classification type of each spell.

* Consequence [float]:  Total influence of a spell, measured by the difference between the sentiment scores of mentions involving the spell i.e. score of mention w/o spell and w/ spell (the Effect is inserted instead of the spell name)

* Sentiment [float]: Sentiment score of the effect of each spell.

* Count [int]: Number of times each spell was used in books 1 - 6.

* Appearance [int]: whether or not a spell was used in book 7.

## Architecture

A simple linear model trained with Linear Regression to make binary classification of spells. Due to insufficient data, [KernelLinearClassifier](https://www.tensorflow.org/api_docs/python/tf/contrib/kernel_methods/KernelLinearClassifier) is used to improve models prediction accuracy.

## Result
Accuracy: 45%

Loss: 0.69

Training time: ~21 sec

## Lesson

Several attempts to improve the model and increase its accuracy, by tuning, was to no avail. I learned that small/insufficient datasets are tough to create a Machine Learning model for with linear classifiers. An architecture such as [Neural Network Random Forest](http://www.public.asu.edu/~swang187/publications/NNRF.pdf) (NNRF) known for training models with small datasets might be employed in the future.
