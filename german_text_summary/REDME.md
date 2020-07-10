# German Text Summary 

## German Text summary with t5
I used the t5 and a german Dataset from a wikipedia crawl to test the t5 model on German Text summary. The rouge score is not as good as the ones from the english cnn daily mail dataset. But because of the different datasets it is not comparable. That is why I'm trying to translate the CNN Daily Mail Dataset.    

## Translate CNN Daily Mail Dataset to German
First I tried to use the t5 model to translate the Dataset, but the translation was not of good quality. So I used another model from Ott, Myle, et al. "Scaling neural machine translation." arXiv preprint arXiv:1806.00187 (2018)., which performed much better on translation.

## Results
The rouge score from the translated german dataset shows that the t5 is not as good as in english. But the performance is 
not to bad. If you look at the [german  sentences](result_german.txt) you mostly think, that they mean the right things,  
but they have problems to get the german language right. But it is interesting to see how a translated dataset is working.
The result rouge scores are close to the wikipedia dataset, which leads to the conclusion, that the lead of the performance 
in english caused by the data domain.  

#### German CNN Daily Mail Rouge Score
- rouge1 = 25.95, 95% confidence [25.37, 26.58]
- rouge2 = 7.55, 95% confidence [7.15, 7.92]
- rougeLsum = 14.76, 95% confidence [14.40, 15.09]

#### German Wikipedia Rouge Score
- rouge1 = 23.55, 95% confidence [22.47, 24.62]
- rouge2 = 6.20, 95% confidence [5.52, 6.85]
- rougeLsum = 16.33, 95% confidence [15.68, 17.12]

#### English CNN Daily Mail Rouge Score
- rouge1 = 38.52, 95% confidence [37.74, 39.39]
- rouge2 = 17.51, 95% confidence [16.71, 18.34]
- rougeLsum = 35.93, 95% confidence [35.14, 36.73]


## References:
- https://pytorch.org/hub/pytorch_fairseq_translation/
- Fecht, Pascal, Sebastian Blank, and Hans-Peter Zorn. "Sequential Transfer Learning in NLP for German Text Summarization." (2019).
### German Text summary Dataset
- https://drive.switch.ch/index.php/s/YoyW9S8yml7wVhN