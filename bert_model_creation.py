import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize  
from nltk.corpus import wordnet
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import BertTokenizer, TFBertModel
import random
import numpy as np

class TextClassificationModel:

    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2
        
    def data_extract(self):
        self.df = pd.concat([self.df1, self.df2])
        return self.df

    def synonym_replacement(self, text, n=1):
        words = word_tokenize(text)
        augmented_texts = [text]

        for _ in range(n):
            new_words = list(words)
            for i, word in enumerate(words):
                synsets = wordnet.synsets(word) 
                if synsets:
                    synonym = random.choice(synsets).lemmas()[0].name()  
                    new_words[i] = synonym
            augmented_texts.append(' '.join(new_words))

        return augmented_texts

    def augment_data(self):

        augmented_data = []
        minority_classes = ["emergency care", "urgent care", "office visits", "mental health"]

        for row in self.df.itertuples():
            label = row.Sydney_Category
            text = row.Service_Name
            feature = row.Category_Name
            
            augmented_data.append((label, text, feature))
            
            if label in minority_classes:
                augmented_texts = self.synonym_replacement(text, n=2)
                for augmented_text in augmented_texts:
                    augmented_data.append((label, augmented_text, feature))

        return pd.DataFrame(augmented_data, columns=self.df.columns)


    def preprocess(self):
        
        self.df = self.data_extract()
        print("printing the shape of dataframe before augmenting",self.df.shape)
        
        # Lowercase
        self.df["Category Name"] = self.df["Category Name"].str.lower()
        self.df["Service Name"] = self.df["Service Name"].str.lower()
        self.df["Sydney Category"] = self.df["Sydney Category"].str.lower()

        # Augment data
        self.df = self.augment_data()
        print("printing the shape of dataframe after augmenting",self.df.shape)
        
        # Create combined column
        self.df["combined_row"] = self.df["Service Name"] + " " + self.df["Category Name"]

        # Split data
        self.X = self.df["combined_row"]
        self.y = self.df["Sydney Category"]
        
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    
    def create_bert_embeddings(self):
      
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = TFBertModel.from_pretrained("bert-base-uncased")

    def tokenize_bert_embeddings(self, texts):
        inputs = self.tokenizer(texts, max_length=40, truncation=True, padding="max_length", return_tensors="tf")
        outputs = self.bert_model(**inputs)
        embeddings = outputs.last_hidden_state
        embeddings = tf.reduce_mean(embeddings, axis=1)
        return embeddings

    def create_model(self):
        
        # Create embeddings
        self.X_train_embeddings = self.tokenize_bert_embeddings(self.X_train)
        self.X_test_embeddings = self.tokenize_bert_embeddings(self.X_test)

        # Create tensors
        self.X_train_tensors = tf.convert_to_tensor(self.X_train_embeddings)
        self.X_test_tensors = tf.convert_to_tensor(self.X_test_embeddings)

        self.y_train_categorical = to_categorical(self.y_train)
        self.y_test_categorical = to_categorical(self.y_test)

        self.y_train_tensors = tf.convert_to_tensor(self.y_train_categorical)
        self.y_test_tensors = tf.convert_to_tensor(self.y_test_categorical)

        # Build model
        self.model = keras.Sequential([
            layers.Flatten(input_shape=self.X_train_tensors.shape[1:]),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'), 
            layers.Dense(self.y_train_tensors.shape[1], activation='softmax')  
        ])

        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self):
        self.model.fit(self.X_train_tensors, self.y_train_tensors, epochs=5, batch_size=32)

    def evaluate(self):
        print(self.model.evaluate(self.X_train_tensors, self.y_train_tensors))
        print(self.model.evaluate(self.X_test_tensors, self.y_test_tensors))

class ModelPredictor():
    
    def __init__(self, model,label_encoder):
        self.model = model
        self.label_encoder = label_encoder
        
    def predict(self, X_new):
        #tfidf_vectorizer= TfidfVectorizer()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        X_new_embeddings = self.tokenizer(X_new, max_length=40, truncation=True, padding="max_length", return_tensors="tf")
        X_new_tensors=tf.convert_to_tensor(X_new_embeddings)
        predictions = self.model.predict(X_new_tensors)
        y_pred = np.argmax(predictions,axis=1)
        y_pred_text = self.label_encoder.inverse_transform(y_pred)
        print("predicting the y_pred_text",y_pred_text)
        
        return y_pred_text