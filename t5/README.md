# Text summarization with the Pretrained T5  

## Pretrained T5  
We will try out the T5 pretrained Transformer from Google. (https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) They say it's the new a "Shared Text-To-Text Framework" for NLP which should explore the limits of transfer learning. They trained the T5 with the C4 dataset, which is a unlabeled dataset and a cleaned version of Common Crawl that is two orders of magnitude larger than Wikipedia (https://www.tensorflow.org/datasets/catalog/c4).

There are multiple sized versions of the T5. The biggest one has 11B Parameters, which is a lot. The large version of BERT(https://github.com/google-research/bert) has 340M Parameters. It's to big to train on the free TPU from colab, So we will use the 3B Pretrained Model.


## Dataset for text summarization
We will try out the CNN DailyMail Dataset. It is the most used Dataset for text summarization.

## Implementation
There is a Tutorial Notebook on Colab which everyone can use for free.
<a href="https://colab.research.google.com/github/google-research/text-to-text-transfer-transformer/blob/master/notebooks/t5-trivia.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

They programmed a nice Envirement to try out all the tfds Datasets(https://www.tensorflow.org/datasets). We are lucky, the CNN DailyMail Dataset is one of them. So we change the code a bit to fit the new task and finetune the model.
