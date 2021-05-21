#import libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from wordcloud import WordCloud, STOPWORDS
import gensim
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)

#load test and train datasets
df_train = pd.read_csv("C:\\Users\\user\\Documents\\nlp project\\train.csv")
df_test = pd.read_csv("C:\\Users\\user\\Documents\\nlp project\\test.csv")
print(df_train.head())

print(df_train.info())

df_train.drop(['keyword','location'],axis=1,inplace=True)
df_test.drop(['keyword', 'location'], axis= 1, inplace = True)

print(df_train.head())

#look through a few non-disaster tweets
print(df_train[df_train['target']==0].head())
#look through a few disaster tweets
print(df_train[df_train['target']==1].head())

length_train = df_train['text'].str.len()
length_test = df_test['text'].str.len()
plt.hist(length_train,bins=20,label = 'train_text')
plt.hist(length_test,bins=20,label='test_text')
plt.legend()
plt.show()

#label and combine train and test datasets
df_train['data_source'] = 'df_train'
df_test['data_source'] = 'df_test'
data = pd.concat([df_train,df_test],ignore_index = True)

#create function to remove patterns from text
def remove_pattern(input_txt,pattern):
    r=re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt

v_remove_pattern = np.vectorize(remove_pattern)

#remove twitter handles 
data['clean_text'] =v_remove_pattern(data['text'],'@[/w]*')
#remove numbers, puctuations anmd other special chracacters
data['clean_text'] = data['clean_text'].str.replace('[^a-zA-Z#]',' ')
#remove short words
data['clean_text'] = data['clean_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
print(data.head())

#tokenize text in the dataframe
tokenized_text = data['clean_text'].apply(word_tokenize) 
print(data.head())

#lemmatizing
lemmatizer = WordNetLemmatizer()
lemmatized_text = tokenized_text.apply(lambda x: [lemmatizer.lemmatize(i)
                                                  for i in x])

for i in range (len(lemmatized_text)):
    lemmatized_text[i] = ' '.join(lemmatized_text[i])
data['clean_text'] = lemmatized_text

print(data.head())

#generate word clouds for all words, normal words and disaster-related words
all_words = ' '.join([text for text in data['clean_text']])
wordcloud = WordCloud(width=800, height = 500, max_font_size=110).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

normal_words = ' '.join([text for text in data['clean_text'][data['target']==0]])
wordcloud = WordCloud(width=800, height = 500, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

disaster_words = ' '.join([text for text in data['clean_text'][data['target']==1]])
wordcloud = WordCloud(width=800, height = 500, max_font_size=110).generate(disaster_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# function to collect hashtags
def hashtag_extract(x):   
            hashtags = []    

            # Loop over the words in the tweet    
            for i in x:        
                ht = re.findall(r"#(\w+)", i)        
                hashtags.append(ht)     
            return hashtags

# extracting hashtags from regular tweets 
HT_regular = hashtag_extract(data['clean_text'][data['target'] == 0]) 

# extracting hashtags from offensive tweets 
HT_negative = hashtag_extract(data['clean_text'][data['target'] == 1]) 

# unnesting list 
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                 'Count': list(a.values())})

#selecting top 20 most frequent hashtags
d = d.nlargest(columns = 'Count',n=20)
'''plt.figure(figsize=(16,5))
ax = sns.barplot(x='Hashtag', y='Count', data = d)
ax.set(ylabel = 'Count')
plt.show()'''

b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()),
                 'Count': list(b.values())})

#selecting top 20 most frequent hashtags
e = e.nlargest(columns = 'Count',n=20)
'''plt.figure(figsize=(16,5))
ax = sns.barplot(data = e, x='Hashtag', y='Count')
plt.show()'''

tfidf_vectorizer = TfidfVectorizer(max_df = 0.90, min_df= 2,
                                  max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(data['clean_text'])
tfidf.shape

c_vectorizer = CountVectorizer(max_df = 0.90, min_df = 2,
                                max_features = 1000, stop_words = 'english')
cou = c_vectorizer.fit_transform(data['clean_text'])
cou.shape

#split the vectorized text samples
train_tfidf = tfidf[:7613,:]
test_tfidf= tfidf[7613:,:]

X_train, X_test, y_train, y_test = train_test_split(train_tfidf, df_train['target'],test_size=0.3)

#using xgboost classification library to model the training data
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
pred = xgb.predict_proba(X_test)
pred_int = pred[:,1]>=0.3
pred_int = pred_int.astype(np.int)

from sklearn.metrics import f1_score
f1_score(y_test,pred_int)

test_pred= xgb.predict_proba(test_tfidf)
test_pred_int = test_pred[:,1]>=0.3
test_pred_int = test_pred_int.astype(np.int)
df_test['target'] = test_pred_int
submission = df_test[['id','target']]
submission.to_csv('submission.csv',index=False)

