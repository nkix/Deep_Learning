import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

# remove commonly used words like 'he', 'she' ....
nltk.download('stopwords')

"""words = nltk.stopwords.words('english')

pd.read_csv('C:/Users/Nrx03/Desktop/科研/as3/bbc-text.csv', encoding='utf-8')

data = pd.read_csv('C:/Users/Nrx03/Desktop/科研/as3/bbc-text.csv')

data.columns = ['category', 'text']
data['category'] = data['category'].str.split()
data['text'] = data['text'].str.split()

data['category'] = data['category'].apply(lambda x: [item for item in x if item not in words])
data['text'] = data['text'].apply(lambda x: [item for item in x if item not in words])

print(data)
"""

