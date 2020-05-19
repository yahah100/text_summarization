import pandas as pd

import tensorflow_datasets as tfds
import tensorflow as tf


class TfToCsv:

    def __init__(self) -> None:
        super().__init__()

        cnn_dailymail = tfds.load(name="cnn_dailymail")

        self.train_tfds = cnn_dailymail['train']
        self.test_tfds = cnn_dailymail['test']
        self.val_tfds = cnn_dailymail['validation']

    @staticmethod
    def normalize_text(text):
        """Lowercase and remove quotes from a TensorFlow string."""
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text, "'(.*)'", r"\1")
        return text

    def map_func(self, features):
        article_text = self.normalize_text(features["article"])
        highlights_text = self.normalize_text(features['highlights'])

        return article_text.numpy().decode('UTF-8'), highlights_text.numpy().decode('UTF-8')

    def tfds_to_numpy(self, ds):
        new_ds = pd.DataFrame(columns=["articles", "highlights"])
        for i, item in enumerate(ds):
            article, higlights = self.map_func(item)
            new_ds = new_ds.append({"articles": article, "highlights": higlights}, ignore_index=True)
        return new_ds

    @staticmethod
    def save_to_csv(ds, name):
        ds.to_csv(str("data/" + name + ".tsv"), sep="\t", index=False)

    @staticmethod
    def load_csv(name):
        return pd.read_csv(str("data/" + name + ".tsv"), sep="\t")

    def create_new_csv(self):
        test_df = self.tfds_to_numpy(self.test_tfds)
        val_df = self.tfds_to_numpy(self.val_tfds)
        train_df = self.tfds_to_numpy(self.train_tfds)

        self.save_to_csv(test_df, "test")
        self.save_to_csv(train_df, "train")
        self.save_to_csv(val_df, "val")
        return train_df, val_df, test_df

    def load_data(self):
        train_df = self.load_csv("train")
        test_df = self.load_csv("test")
        val_df = self.load_csv("val")
        return train_df, val_df, test_df
