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

## Thematic clustering
