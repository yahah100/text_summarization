# German Text Summary 

## German Text summary with t5
I used the t5 and a german Dataset from a wikipedia crawl to test the t5 model on German Text summary. The rouge score is not as good as the ones from the english cnn daily mail dataset. But because of the different datasets it is not comparable. That is why I'm trying to translate the CNN Daily Mail Dataset.    

## Translate CNN Daily Mail Dataset to German
First I tried to use the t5 model to translate the Dataset, but the translation was not of good quality. So I used another model from Ott, Myle, et al. "Scaling neural machine translation." arXiv preprint arXiv:1806.00187 (2018)., which performed much better on translation.



## References:
- https://pytorch.org/hub/pytorch_fairseq_translation/
- Fecht, Pascal, Sebastian Blank, and Hans-Peter Zorn. "Sequential Transfer Learning in NLP for German Text Summarization." (2019).
### German Text summary Dataset
- https://drive.switch.ch/index.php/s/YoyW9S8yml7wVhN