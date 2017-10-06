"""
    A model for predicting whether or not a spell in the Harry Potter book series was used in the seventh book.

    This project was inspired by Code2040's 2017 Fellow application. The datasets on spells was provided by Code2040 (in two forms: spells & mentions) and modified by me to fit the training method used (Linear Regression).

    The datasets were preprocessed in Ruby (I have reasons why. The code will be transformed to Python in the future.) The datasets has the following features:

    Classification -> [charm, curse, jinx, spell]: Classification type of each spell.

    Consequence -> int:  Total influence of a spell, measured by the difference between the sentiment scores of mentions involving the spell i.e. score of mention w/o spell and w/ spell (the Effect is inserted instead of the spell name)

    Sentiment -> int: Sentiment score of the effect of each spell.

    Count -> int: Number of times each spell was used in books 1 - 6.

    Appearance -> int: whether or not a spell was used in book 7.

    Olu Gbadebo
    Oct. 1, 2017
"""

import pandas as pd
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import csv
import tempfile
import argparse
import sys
import time

training_data = pd.read_json('training_data.json', orient='records')
testing_data = pd.read_json('testing_data.json', orient='records')

train_labels = training_data["Appearance"]
test_labels = testing_data["Appearance"]

# feature columns
classification = tf.feature_column.categorical_column_with_vocabulary_list(
    "Classification", ["Charm", "Curse", "Jinx", "Spell"])
spell = tf.feature_column.categorical_column_with_hash_bucket(
    "Spell", hash_bucket_size=100)
sentiment = tf.feature_column.numeric_column("Sentiment")
sentiment_buckets = tf.feature_column.bucketized_column(
    sentiment, boundaries=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
count = tf.feature_column.numeric_column("Count")
consequence = tf.feature_column.numeric_column("Consequence")

# TODO: calc stddev for each feature
# Kernel mappers
classification_km = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
  input_dim=4, output_dim=100, stddev=5.0, name='rffm')
spell_km = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
  input_dim=92, output_dim=1000, stddev=5.0, name='rffm')

kernel_mappers = {
    classification: [classification_km],
    spell: [spell_km]
}

# sparse column to compensate for insufficient  data
sparse_classification = tf.contrib.layers.sparse_column_with_hash_bucket("Classification", 4)
sparse_spell = tf.contrib.layers.sparse_column_with_hash_bucket("Spell", 100)
sparse_count = tf.contrib.layers.sparse_column_with_integerized_feature("Count", 1000)

base_columns = [
    sparse_classification, sparse_spell, sentiment_buckets, sparse_count, consequence
]
crossed_columns = [
     tf.feature_column.crossed_column(
     ["Sentiment", "Consequence"],
     hash_bucket_size=1000)
]

def input_fn(data, num_epochs, shuffle):
  """Input builder function."""
  data.dropna(how="any", axis=0)
  labels = data["Appearance"]
  return tf.estimator.inputs.pandas_input_fn(
      x=data,
      y=labels,
      batch_size=1,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=1)

# Uncomment next line for debugging in CLI
# hooks = [tf_debug.LocalCLIDebugHook()]

def build_estimator(model_dir):
    """Build an estimator."""
    return tf.contrib.kernel_methods.KernelLinearClassifier(
    feature_columns=base_columns + crossed_columns,
    model_dir=model_dir,
    optimizer=tf.train.FtrlOptimizer(
      learning_rate=0.0001,
      l2_regularization_strength=0.0001)
     )

def train_and_eval(model_dir, train_steps):
    """Train and evaluate the model."""

    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    m = build_estimator(model_dir)

    # add 'hooks = hooks' to debug training process
    m.fit(
    input_fn=input_fn(training_data, num_epochs=None, shuffle=True),
    steps=train_steps)

    m.evaluate(
    input_fn=input_fn(testing_data, num_epochs=1, shuffle=False),
    steps=None)

    prediction = m.predict_classes(
    input_fn=input_fn(testing_data, num_epochs=1, shuffle=False))

    print(list(prediction))

FLAGS = None

def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.train_steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
    "--model_dir",
    type=str,
    default="",
    help="Base directory for output models."
      )
    parser.add_argument(
    "--train_steps",
    type=int,
    default=2000,
    help="Number of training steps."
      )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
