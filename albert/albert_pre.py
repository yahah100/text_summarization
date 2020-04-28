import os
import pandas as pd
import numpy as np

from flair.data import Sentence
from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings

from my_sentence_piecer import MySentencePiecer
from segtok.segmenter import split_single


class AlbertPre:

    def __init__(self, MAX_WORD_N=150, MAX_SENT_N=30, MAX_WORD_SENT_N=300, alber_model="albert-base-v2") -> None:
        super().__init__()
        albert = BertEmbeddings(bert_model_or_path=alber_model)
        self.albert_embedding = DocumentPoolEmbeddings([albert])
        self.MAX_WORD_N = MAX_WORD_N
        self.MAX_SENT_N = MAX_SENT_N
        self.MAX_WORD_SENT_N = MAX_WORD_SENT_N

        self.sentence_piecer = MySentencePiecer()

    def get_embedding(self, sentence):
        sent = Sentence(sentence)
        self.albert_embedding.embed(sent)
        return sent.get_embedding()

    @staticmethod
    def split_in_sentences(text):
        return split_single(text)

    @staticmethod
    def load_csv(name):
        return pd.read_csv(str("../data/" + name + ".tsv"), sep="\t")

    def load_data(self):
        train_df = self.load_csv("train")
        test_df = self.load_csv("test")
        val_df = self.load_csv("val")
        return train_df, val_df, test_df

    def embed_sentences(self, sentences):
        arr_embedding = np.zeros((self.MAX_SENT_N, 3072))
        for i, sentence in enumerate(sentences):
            if len(sentence) > 0 and i < self.MAX_SENT_N:
                x = self.get_embedding(sentence[:self.MAX_WORD_SENT_N])
                x = x.to('cpu').detach().numpy()
                arr_embedding[i] = x

        return arr_embedding

    def compute_and_save_df(self, ds, name, size_chunks=20000):
        len_ds = len(ds)
        chunks = int(len_ds / size_chunks)
        for chunk in range(chunks+1):
            part_df = ds.iloc[:][(chunk*size_chunks):(size_chunks + chunk*size_chunks)]
            part_len = len(part_df)

            article_np = np.zeros((part_len, 30, 3072))

            highlight_list = []
            n_highlight_list = []
            n_article_list = []

            for i, (article, highlight) in ds.iterrows():
                article_sent = self.split_in_sentences(article)
                n_areticle = len(article_sent)
                article_np[i] = self.embed_sentences(article_sent)

                highlight_ids = np.array(self.sentence_piecer.get_ids_from_vocab(highlight))[:self.MAX_WORD_N]

                highlight_list.append(highlight_ids)
                n_highlight_list.append(highlight_ids.shape[0])
                n_article_list.append(n_areticle)

                if (i % 1000) == 0:
                    print("computed [%d/%d/%d]" % (chunk, i, len_ds))

            path = "../data/%s" % (name)
            if not os.path.exists(path):
                os.mkdir(path)

            chunks_str = ""
            if chunk != 0:
                chunks_str = "_" + str(chunk)
            np.save(str(path + "/article" + chunks_str + ".npy"), article_np)
            np.save(str(path + "/n_highlights" + chunks_str + ".npy"), n_highlight_list)
            np.save(str(path + "/n_articles" + chunks_str + ".npy"), n_article_list)
            np.save(str(path + "/highlights" + chunks_str + ".npy"), highlight_list)

    @staticmethod
    def load_np_files(name):
        path = "../data/%s" % name

        article_np = np.load(str(path + "/article" + ".npy"), allow_pickle=True)
        n_highlights = np.load(str(path + "/n_highlights" + ".npy"), allow_pickle=True)
        n_articles = np.load(str(path + "/n_articles" + ".npy"), allow_pickle=True)
        highlights = np.load(str(path + "/highlights" + ".npy"), allow_pickle=True)

        return article_np, n_articles, n_highlights, highlights


if __name__ == "__main__":
    # execute only if run as a script
    alpert_pre = AlbertPre()
    _, val_df, _ = alpert_pre.load_data()

    alpert_pre.compute_and_save_df(val_df, "val")
