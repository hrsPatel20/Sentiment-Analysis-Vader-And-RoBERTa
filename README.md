# Sentiment Analysis on Mobile Reviews

This project explores sentiment analysis on mobile reviews using two different models: VADER (Valence Aware Dictionary and sEntiment Reasoner) and RoBERTa (A Robustly Optimized BERT Pretraining Approach). The goal is to compare the performance of these models in analyzing the sentiment of text data.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
  - [VADER](#vader)
  - [RoBERTa](#roberta)
- [Setup](#setup)
- [Results](#results)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [License](#license)

## Introduction

Sentiment analysis is a common natural language processing task that involves classifying text into sentiment categories such as positive, negative, or neutral. In this project, we perform sentiment analysis on mobile reviews to determine the overall sentiment expressed by users.

## Dataset

The dataset used in this project contains mobile reviews collected from various sources. Each review is labeled with a sentiment score or category. The dataset is preprocessed to remove noise and prepare it for model input.



## Models

### VADER

VADER is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. It is known for its effectiveness in text that is shorter, like mobile reviews, and for being quick to implement.

### RoBERTa

RoBERTa is a transformer-based model optimized for a variety of NLP tasks. It is a powerful model for understanding context and nuances in text, making it suitable for more complex sentiment analysis tasks.

## Setup

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Libraries: `transformers`, `vaderSentiment`, `pandas`, `numpy`, `matplotlib`

### Installation

    ```
    pip install transformers vaderSentiment pandas numpy matplotlib


## Results
The comparison between the VADER and RoBERTa models is detailed in the notebook. The evaluation metrics include accuracy, precision, recall, and F1 score. The results show how each model performs on the dataset of mobile reviews.

VADER: Provides quick results and is effective for shorter texts.
RoBERTa: Offers a deeper understanding of the text, capturing nuances that VADER might miss.

## Conclusion
This project highlights the strengths and weaknesses of both VADER and RoBERTa in the context of sentiment analysis on mobile reviews. While VADER is faster and works well with shorter texts, RoBERTa provides a more detailed analysis at the cost of increased computational complexity.

## How to Run
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/sentiment-analysis-mobile-reviews.git

2. Navigate to the project directory:

   ```
   cd sentiment-analysis-mobile-reviews

3. Open the Jupyter Notebook:

   ```
   jupyter notebook Sentiment_Analysis.ipynb

4. Run all cells in the notebook to perform the analysis.


   
   

   



