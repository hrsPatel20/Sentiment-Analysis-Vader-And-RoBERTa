#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


import nltk


#read data
#data = mobile Reviews 
#source = https://drive.google.com/file/d/1NYdZoMJvBWuCejMX28pVRVfMyOe1GhnZ/view

df = pd.read_csv(r'C:/Users/mitpa/OneDrive/Desktop/harsh/mobile reviews.csv',encoding='unicode_escape')
print(df.shape)


# In[2]:


#sorting
df = df.head(500)
print(df.shape)
df.head()


# In[3]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()


# In[4]:


ex = "Ram is good boy!"
ex1 = sia.polarity_scores(ex)
print(ex1)


# In[5]:


# Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    body = row['body']
    myid = row['Id']
    res[myid] = sia.polarity_scores(body)


# In[6]:


res


# In[7]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')
vaders.head()


# In[ ]:





# In[8]:


def sentiment_vader(body):
    overall_polarity = sia.polarity_scores(body)
    if overall_polarity['compound'] > 0.5:
        return "Positive"
    elif overall_polarity['compound'] <-0.5:
        return "Negative" 
    else:
        return "Neutral"
vresults_df = pd.DataFrame(vaders)
vresults_df['Sentiment_vader'] = vresults_df['body'].apply(lambda x: sentiment_vader(x))
vresults_df.head(20)



# In[9]:


vader_df = pd.DataFrame(vresults_df)

def remove_unnecessary_columns(vader_df, columns_to_remove):
    for column in columns_to_remove:
        if column in vader_df.columns:
            vader_df.drop(column, axis=1, inplace=True)
        else:
            break

    return vader_df

vader_df =vader_df[['Id','neg','pos','neu','compound','asin','name','date','verified','title','body','helpfulVotes','Unnamed: 9','Unnamed: 10','Unnamed: 11','Sentiment_vader']]

columns_to_remove = ['verified','helpfulVotes','Unnamed: 9','Unnamed: 10','Unnamed: 11']

vader_df = remove_unnecessary_columns(vader_df, columns_to_remove)
vader_df.head(500)


# In[ ]:





# In[10]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[11]:


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[12]:


example = df['body'][50]
print(example)


# In[13]:


# VADER results on example
print(example)
sia.polarity_scores(example)


# In[14]:


example = df['body'][50]
print(example)


# In[15]:


# Run for Roberta Model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)


# In[16]:


def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict


# In[17]:


res1 = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        body = row['body']
        myid = row['Id']
        vader_result = sia.polarity_scores(body)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(body)
        both = {**vader_result_rename, **roberta_result}
        res1[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')


# In[18]:


res1


# In[19]:


results_df = pd.DataFrame(res1).T

results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')


# In[20]:


results_df.head()


# In[21]:


dresults_df = pd.DataFrame(results_df)
dresults_df.head()


# In[22]:


def sentiment_vader(body):
    overall_polarity = sia.polarity_scores(body)
    if overall_polarity['compound'] >= 0.05:
        return "Positive"
    elif overall_polarity['compound'] <= -0.05:
        return "Negative" 
    else:
        return "Neutral"


# In[23]:


dresults_df['Sentiment_vader'] = results_df['body'].apply(lambda x: sentiment_vader(x))
dresults_df.head(500)


# In[24]:


vader_df = pd.DataFrame(dresults_df)

def remove_unnecessary_columns(vader_df, columns_to_remove):
    for column in columns_to_remove:
        if column in vader_df.columns:
            vader_df.drop(column, axis=1, inplace=True)
        else:
            break

    return vader_df

vader_df =vader_df[['Id','vader_neg','vader_pos','vader_neu','vader_compound','roberta_neg','roberta_neu','roberta_pos','asin','name','date','verified','title','body','helpfulVotes','Unnamed: 9','Unnamed: 10','Unnamed: 11','Sentiment_vader']]

columns_to_remove = ['roberta_neg','vader_neg','vader_pos','vader_neu','vader_compound','roberta_neu','roberta_pos','verified','helpfulVotes','Unnamed: 9','Unnamed: 10','Unnamed: 11']

vader_df = remove_unnecessary_columns(vader_df, columns_to_remove)
vader_df.head(500)



        
           
            
            
          




# In[ ]:





# In[ ]:





# In[25]:


#slicing roberta results
new_df = pd.DataFrame(results_df)

final_result = new_df[['roberta_neg','roberta_neu','roberta_pos']]
final_result.head()


# In[26]:


#max value id

MaxValues = final_result.idxmax(axis=1)
final_df = pd.DataFrame(MaxValues)
final_df.head()


# In[27]:


#replace
final_df.replace(['roberta_neg', 'roberta_pos','roberta_neu'], ['Negative', 'Positive','Neutral'], inplace=True)

final_df.head()


# In[28]:


ddresults_df = pd.DataFrame(results_df)
ddresults_df["Sentiment_RoBERTa"] = final_df
ddresults_df.head()
#now remove the unnecessary columns

roberta_df = pd.DataFrame(ddresults_df)

def remove_unnecessary_columns(roberta_df, columns_to_remove):
    for column in columns_to_remove:
        
        if column in roberta_df.columns:
            roberta_df.drop(column, axis=1, inplace=True)
        else:
            break

    return roberta_df

roberta_df =roberta_df[['Id','vader_neg','vader_pos','vader_neu','vader_compound','roberta_neg','roberta_neu','roberta_pos','asin','name','date','verified','title','body','helpfulVotes','Unnamed: 9','Unnamed: 10','Unnamed: 11','Sentiment_RoBERTa']]

columns_to_remove = ['roberta_neg','vader_neg','vader_pos','vader_neu','vader_compound','roberta_neu','roberta_pos','verified','helpfulVotes','Unnamed: 9','Unnamed: 10','Unnamed: 11']

roberta_df = remove_unnecessary_columns(roberta_df, columns_to_remove)
roberta_df.head(500)


# In[ ]:





# In[29]:


compare_df = pd.DataFrame(vader_df)
compare_df['Sentiment_RoBERTa'] = final_df
compare_df.head(500)

