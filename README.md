# Machine Translation and  Multilinguality in Text classification.
Team Members: Nesma Mahmoud B87771, Mahmoud Kamel B87770
This project is related to at the University of Tartu, Institute of Computer Science. 
Our project consists of two main parts:
*  Handling multilinguality in text classification
*  Expanding the available data with Round-trip-translation

# Datset: 
[multilingual-text-categorization-dataset](https://github.com/valeriano-manassero/multilingual-text-categorization-dataset)
This data set contains blog posts in 32 Language categorized into 45 Category. 
* Categories: ['advertising', 'agriculture', 'animation', 'arts_and_crafts',
       'entertainment', 'astrology', 'vehicles', 'games',
       'books_and_literature', 'business', 'gambling', 'jobs', 'clothing',
       'comic_books', 'dating', 'education', 'adult', 'food', 'health',
       'hobbies_and_interests', 'humor', 'illegal_content', 'investing',
       'jewelry', 'logistics', 'marketing', 'movies', 'music', 'hacking',
       'media', 'finance', 'pets', 'politics', 'religion',
       'sci_fi_and_fantasy', 'science', 'shopping', 'society', 'sports',
       'tech', 'teens', 'television', 'travel', 'under_construction',
       'weather']
       
* Languages: ['english', 'albanian', 'arabic', 'bulgarian', 'chinese',
       'croatian', 'czech', 'danish', 'dutch', 'estonian', 'finnish',
       'french', 'german', 'greek', 'hebrew', 'hungarian', 'icelandic',
       'italian', 'japanese', 'korean', 'lithuanian', 'norwegian',
       'polish', 'portuguese', 'romanian', 'russian', 'serbian',
       'slovenian', 'spanish', 'swedish', 'turkish', 'ukrainian'],

# Project Scope: 
# 1. Handling multilinguality in text classification:
In this part we will try three different ways for multilingual text classification, and compare between them. The three differet methods are:
1. Comparing Joint multilingual approach: we classify all of the languages together with single classification system (can be also ensemble of multilingual models)
2. Joint translated monolingual: all languages are translated into one super-language - prolly english - and then classified all together.
3. multiple monolingual classification approach: each language has a separate classification system trained to it.

# 2. Expanding the available data with Round-trip-translation:
This part involves testing how to best leverage the increased diversity that RT-translation brings to the data.

# Frameworks:
1- Keras  
2- AllenNlp  
3- FLAIR 

# Experiments:
1- Keras Experiments and Results can be found [here](https://github.com/nesmaAlmoazamy/Handling_Multilinguality/blob/master/Keras/ALL_PHASES.ipynb), except there are some experiments that was run over the server as: 
* The Translation model results and code which can be found [here](https://github.com/nesmaAlmoazamy/Handling_Multilinguality/tree/master/Keras/Joint_Translated_Monolingual) 
 ** The Datasets Translation to English Using IBM can be found [here](https://github.com/nesmaAlmoazamy/Handling_Multilinguality/blob/master/Keras/TranslationUsingIBMWatson.ipynb)
 
 ** The Languages after being Translated to lenguage can be found [here](https://github.com/nesmaAlmoazamy/Handling_Multilinguality/blob/master/Keras/Super%20Language%20after%20translation.ipynb)  
 ** we also Tried using Google Translation, but we reached the limit, and we couldn't find a way around that, That's why we moved to IBM Watson for the translations. which also have a limit but we managed to work around it.
* The Joint Multilingual model results and code which can be found [here](https://github.com/nesmaAlmoazamy/Handling_Multilinguality/tree/master/Keras/Joint_Multilingual_Results) 


2- AllenNlp Results are Included in this [notebook]( https://github.com/nesmaAlmoazamy/Handling_Multilinguality/blob/master/AllenNlp/ALL_PHASES_AllenNlp.ipynb)
For Running AllenNlp from Configuration file, we need those files: 
* DataReader Class can be found [here](https://github.com/nesmaAlmoazamy/Handling_Multilinguality/blob/master/AllenNlp/data-reader.py)
* Predictor Class can be found [here](https://github.com/nesmaAlmoazamy/Handling_Multilinguality/blob/master/AllenNlp/predictor.py)
* Model Class can be found [here](https://github.com/nesmaAlmoazamy/Handling_Multilinguality/blob/master/AllenNlp/LstmClassifier.py)
* Configuration file can be found [here](https://github.com/nesmaAlmoazamy/Handling_Multilinguality/blob/master/AllenNlp/ArticlesClassification.jsonnet)

3-Flair Experiments Results: 
* We used Bert through Flair Framework, but it failed because Bert can only work for data sequences that are less than 512, and our dataset has articles with more number of sequences. you can see the experiment [here](https://github.com/nesmaAlmoazamy/Handling_Multilinguality/blob/master/Flair/FlairBertEnglishFAILED.ipynb)
* After that we tried decreasing the number of tokens per each article which can be found [here](https://github.com/nesmaAlmoazamy/Handling_Multilinguality/blob/master/Flair/FlairBertEnglishDecreaseNumberOfTokensTo50.ipynb) but it resulted in a bad results also so we decided not to continue with FLAIR

* We used FLAIR stacked embeddings for english classification. we found that it consumes alot of resources but as a POC we trained it over a subset of the english dataset. and it worked well for this subset. you can find the experiment here: This is the [POC](https://github.com/nesmaAlmoazamy/Handling_Multilinguality/blob/master/Flair/FlairEnglishClassificationPOC.ipynb)
* After that we Trained it over all the English dataset, code [here]( https://github.com/nesmaAlmoazamy/Handling_Multilinguality/blob/master/Flair/FlairEnglishClassificationToBeRunOverTheCluster.ipynb)
* But it resulted in a very strange results, which can be seen from [here]( https://github.com/nesmaAlmoazamy/Handling_Multilinguality/blob/master/Flair/result_flair.txt)

Visualization code of the graphs that are used in the blog post can be found [here](https://github.com/nesmaAlmoazamy/Handling_Multilinguality/blob/master/Visualization/Visualizations.ipynb)

Blog post can be found [here](https://medium.com/@mahmoud.kamel104/machine-translation-and-multilinguality-in-text-classification-6e20ef9dbce8)
