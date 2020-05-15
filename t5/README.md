# Text summarization with the Pretrained T5  

## Pretrained T5  
We will try out the T5 pretrained Transformer from Google. (https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) They say it's the new a "Shared Text-To-Text Framework" for NLP which should explore the limits of transfer learning. They trained the T5 with the C4 dataset, which is a unlabeled dataset and a cleaned version of Common Crawl that is two orders of magnitude larger than Wikipedia (https://www.tensorflow.org/datasets/catalog/c4).

There are multiple sized versions of the T5. The biggest one has 11B Parameters, which is a lot. The large version of BERT(https://github.com/google-research/bert) has 340M Parameters. It's to big to train on the free TPU from colab, So we will use the 3B Pretrained Model.


## Dataset for text summarization
We will try out the CNN DailyMail Dataset. It is the most used Dataset for text summarization.

## Implementation
There are 3 Types of implementations, first the implementation from google, then a tensorflow and pytorch implementation, which are based on the huggingface library. The ones based on the huggingface library have the plus points, that they are better expandable or customizable. 

###  [Google Implemenation](text_summary_with_t5.ipynb) 
They programmed a nice environment to try out all the tfds Datasets(https://www.tensorflow.org/datasets). We are lucky, the CNN DailyMail Dataset is one of them. So we change the code a bit to fit the new task and finetune the model.
<a href="https://colab.research.google.com/github/google-research/text-to-text-transfer-transformer/blob/master/notebooks/t5-trivia.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### [Huggingface Tensorflow Implemenation](t5_tf_huggingface.ipynb) 

### [Huggingface Pytorch Implemenation](t5_pt_huggingface.ipynb) 
