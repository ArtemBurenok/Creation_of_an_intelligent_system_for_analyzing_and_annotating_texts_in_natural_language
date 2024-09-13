# Creation of an intelligent system for analyzing and annotating texts in natural language

The project is about analyzing natural language. Namely semantic analysis, automatic annotation and thematic clustering.

## Content
+ Technologies
+ Semantic analysis
+ Automatic annotation
+ Thematic clustering

## Technologies
Before analyzing the text directly, preprocessing with NLTK was performed. Namely, lemmatization, stemming, stop word removal and tokenization. Embeddings were also created using Word2Vec.

A dictionary approach, specifically VaderSentiment, was chosen to determine the tone of the text. The T5 model was used for automatic annotation. And for thematic clustering KMeans was used.

## Semantic analysis

The tool used in this work is [VADER](https://github.com/brunneis/vader-multi) (Valence Aware Dictionary and sEntiment Reasoner), which is a rule-based dictionary and sentiment analysis tool that is specifically built to identify positive and negative emotions. This library uses the following algorithm.

It is necessary to first build a dictionary of words and expressions in which each element of the dictionary is assigned a tone score. 

After the text to be analyzed is matched with the available dictionary. The tonality of the text is obtained from the tonality of the sentences, which in turn are obtained from the tonality of the words. Also, rules that take into account the context of the words are applied.

At the end, special rules are used to calculate or modify the tone score:
+ checking for words that reinforce the evaluation of the evaluative word (very) or change the evaluation to the opposite (not, no)
+ calculation of the arithmetic mean of evaluations of words in the text or statement
+ negative evaluation of a word combination if it contains at least one negative expression

## Automatic annotation

In this paper, several abstracting approaches have been considered:
+ **TextRank** is one of the oldest and most popular methods of automatic abstracting. The main idea of the method is to represent the text as a graph with the subsequent calculation of the importance of each syntactic unit.
+ **Embeddings** this approach calculates vector embeddings of sentences and computes a measure of closeness (often a cosine distance) between them. The most “close” sentences are then selected and placed in the final annotation.
+ **Greedy algorithm** Greedy algorithms are algorithms whose principle of operation is to make locally optimal solutions at each iteration. Since it is easier to find a local solution than a global one (if it is possible at all), such algorithms have good asymptotics.

  It happens that the process of finding an optimal solution gets stuck at a local point. To invent and prove correctness of a greedy algorithm is often quite a difficult task.
  
  Let us proceed directly to the implementation of the greedy algorithm for text annotation. At the beginning, a random sentence from the text is selected and the similarity metric with the existing annotation is computed for it (it will increase at the first iteration, because at the beginning the annotation is an empty string). After that, also randomly, other sentences are selected and the similarity metric with the desired annotation is computed. If the metric grows when adding sentences, then the sentence is included in the annotation, otherwise the process stops. Thus, a system of sentences with maximum similarity coefficient is obtained.
  
  As can be seen, this type of algorithm is very dependent on the first, randomly selected, sentence. Potentially, the algorithm can choose an unimportant sentence, then the whole annotation will be incorrect, or it will contain important sentences, but in a small amount.
+ **Transformer** The idea of this architecture is based on the attention mechanism. In transformers, the architecture consists of several encoders and decoders. However, the information that an encoder returns will not be represented as a single vector, it will be represented as multiple vectors.

  You can read more about transformers and the attention mechanism [here](https://arxiv.org/abs/1706.03762).

## Thematic clustering
