# My learning try on Transformers

## Text Summary with Flair Embedding
First I split up the sentences of the CNN Daily Mail Dataset to compute the flair Embedding. Then I saved them on the hard drive, so that I do not have to computed it while training.    

### Model
I used Pytorch to build a transformer which has the precomputed embedding and the tokenized articles as Input. I build one Part to handle the embedding, which should add up a context per sentence and the other part to handle the article input for the language. To generate a Summary I tried a autoregressive approach.   

### Result
My model did not work well. It overfitted after some time and had problems to build english sentences. But it was a good exercise to learn something about transfomers and text summary.  
   