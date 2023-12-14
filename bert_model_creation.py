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
import pickle
from tensorflow.keras.callbacks import EarlyStopping


class TextClassificationModel:

    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2
        # self.tokenizer=None
        # self.bert_model=None
        
    def data_extract(self):
        self.df = pd.concat([self.df1, self.df2])
        self.df["Category Name"] = self.df["Category Name"].str.lower()
        self.df["Service Name"] = self.df["Service Name"].str.lower()
        self.df["Sydney Category"] = self.df["Sydney Category"].str.lower()
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
        minority_classes = ["emergency care", "urgent care", "office visits", "mental health","unknown category"]
        label = self.df["Sydney Category"]
        text = self.df["Service Name"]
        feature=self.df["Category Name"]

        for label,text,feature in zip(label,text,feature):
            augmented_data.append((label, text, feature))
            
            if label in minority_classes:
                augmented_texts = self.synonym_replacement(text, n=1)
                for augmented_text in augmented_texts:
                    #print("augmented_text order is ",augmented_text)
                    augmented_data.append((label, augmented_text, feature))

        return pd.DataFrame(augmented_data, columns=["Sydney Category","Service Name","Category Name"])


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
        print("shape of self.X before splitting",self.X.shape)
        self.y = self.df["Sydney Category"]
        print("shape of self.y before splitting",self.y.shape)
        
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)
        pickle.dump(self.label_encoder, open('label_encoder.pkl', 'wb'))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        print(self.X_train.shape,self.X_test.shape,self.y_train.shape,self.y_test.shape)
    
    def create_bert_embeddings(self):
      
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = TFBertModel.from_pretrained("bert-base-uncased")
        pickle.dump(self.tokenizer, open('tokenizer.pkl', 'wb'))

    def tokenize_bert_embeddings(self, texts):
        inputs = self.tokenizer(texts, max_length=40, truncation=True, padding="max_length", return_tensors="tf")
        outputs = self.bert_model(**inputs)
        embeddings = outputs.last_hidden_state
        embeddings = tf.reduce_mean(embeddings, axis=1)
        return embeddings

    def get_bert_embeddings_in_chunks(self,texts,chunk_size=10):
        embeddings =[]


        for i in range(0,len(texts),chunk_size):
            chunk = texts[i:i+chunk_size]
            print(i)
            # print(chunk)
            chunk_embeddings = self.tokenize_bert_embeddings(chunk)
            embeddings.append(chunk_embeddings)
        return tf.concat(embeddings,axis=0)    

    def create_model(self):
        
        self.X_train=self.X_train.tolist()
        self.X_test =self.X_test.tolist()
        # Create embeddings
        self.X_train_embeddings = self.get_bert_embeddings_in_chunks(self.X_train)
        self.X_test_embeddings = self.get_bert_embeddings_in_chunks(self.X_test)

        # Create tensors
        self.X_train_tensors = tf.convert_to_tensor(self.X_train_embeddings)
        self.X_test_tensors = tf.convert_to_tensor(self.X_test_embeddings)

        self.y_train_categorical = to_categorical(self.y_train)
        self.y_test_categorical = to_categorical(self.y_test)

        self.y_train_tensors = tf.convert_to_tensor(self.y_train_categorical)
        self.y_test_tensors = tf.convert_to_tensor(self.y_test_categorical)
        # Build model
        num_classes=len(self.df["Sydney Category"].unique())
        print("printing of num_classes for the softmax layer",num_classes)

        self.model = keras.Sequential([
            layers.Flatten(input_shape=self.X_train_tensors.shape[1:]),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'), 
            layers.Dense(num_classes, activation='softmax')
        ])
        initial_lr = 0.0001
        opt = keras.optimizers.Adam(learning_rate=initial_lr)
        self.model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        self.model.fit(self.X_train_tensors, self.y_train_tensors, epochs=30, validation_split=0.2,batch_size=32,verbose=1)

    def evaluate(self):
        print(self.model.evaluate(self.X_train_tensors, self.y_train_tensors))
        print(self.model.evaluate(self.X_test_tensors, self.y_test_tensors))

class ModelPredictor():
    
    def __init__(self, model,label_encoder):
        self.model = model
        self.label_encoder = label_encoder
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = TFBertModel.from_pretrained("bert-base-uncased")
        
    # def create_bert_embeddings(self):
      
    #     self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #     self.bert_model = TFBertModel.from_pretrained("bert-base-uncased")


    def tokenize_bert_embeddings(self, texts):
        inputs = self.tokenizer(texts, max_length=40, truncation=True, padding="max_length", return_tensors="tf")
        outputs = self.bert_model(**inputs)
        embeddings = outputs.last_hidden_state
        embeddings = tf.reduce_mean(embeddings, axis=1)
        return embeddings

    def predict(self, X_new):
        embeddings = self.tokenize_bert_embeddings(X_new)
        X_new_tensors=tf.convert_to_tensor(embeddings)
        #print("")
        predictions = self.model.predict(X_new_tensors)
        # y_pred = np.argmax(predictions,axis=1)
        # y_pred_text = self.label_encoder.inverse_transform(y_pred)
        # print("predicting the y_pred_text",y_pred_text)
        
        return predictions