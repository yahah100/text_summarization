# Text summarization 

## Evaluation and Dataset
First we need a metric to evaluate our results. For textsummarisation the rouge score (https://www.aclweb.org/anthology/W04-1013.pdf) is commenly used. This metric counts the matched N-grams. 

Then we need a large dataset which is popular so that we can compare our results. This is fast found and it’s the “cnn daily mail” dataset (https://github.com/abisee/cnn-dailymail).

## Model
After that we need a state of the art language model. The idea is to use a Transformer like Bert to get good results for text summarisation. Now we have a lot to choose from: There is Bert, GPT2, XLNET and many more. There are also higher scaled version like the t5 or even smaller models with state of the art performance like ALBERT or Robert.

### [t5 CNN Daily Mail Summary](t5)
I chose the t5 transformer because it was easy to use and I could use the google colab TPU to finetune a pretrained model which has 3B parameters. I got a better rouge score than a Paper from (https://arxiv.org/pdf/1902.09243.pdf) which is from 2019.

There is also a huggingface implementation which can be used in pytorch and Tensorflow. I implemented both for text summary on the daily mail Dataset. The huggingface implementation gives you more control over the model, how to train it or expand it. 
### [My Transformer](my_transformer_try)
To better learn how text summary works, I tried to build a Transformer with pytorch by myself. I wanted to use a flair Embedding so that I do not have to train from ground up. The results have not been so good, so I went back to the huggingface implementations.

As a Model from flair I wanted try out Albert to see if I could get results like the t5. Albert is a smaller Model, which has similar performance like Bert-Large with 18x fewer parameters and can be trained about 1.7x faster. (https://arxiv.org/abs/1909.11942). But the Albert model is not so easy to use for text summary thats why I used a flair Embedding. 

### [German Text summary](german_text_summary)
The results with the t5 model were amazingly good, so I wanted to try out how good the t5 model handles the German language. I found a German Dataset from a Wikipedia crawl. The results were not comparable to the results from, the English language. Because of that I want to translate the Cnn Daily Mail Dataset to have better comparison.
First I tried to use the t5 model to translate the Dataset, but the translation was not of good quality. So I used another model, which performed much better on translation.
     