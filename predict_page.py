
# add requirements 
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pickle
from pickle import load

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# path 
path = "./models/casual_formal"

# create new features from text 
def preprocess(text):
    df = pd.DataFrame(text,columns=['sentence'],index=[0])
    
    # word count 
    df['word_count'] = df.sentence.apply(lambda x: len(str(x).split(" ")))
    df['char_count'] = df.sentence.apply(lambda x: sum(len(word) for word in str(x).split(" ")))

    # average word length
    df['avg_word_length'] = df['char_count'] / df['word_count']

    # create a list of formal pronouns
    formal_prons = [
        "we","they", "their", "themselves", "us", "our", "ourselves", "ours", "it", "its", "itself"
    ]
    informal_prons = [
        "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself", "yourselves"
    ]
    # count if matching formal pronouns in list 
    df['formal_pron'] = df.sentence.apply(lambda x: sum(x.count(word) for word in formal_prons))
    # count if matching informal pronouns in list 
    df['informal_pron'] = df.sentence.apply(lambda x: sum(x.count(word) for word in informal_prons))

    contraction_list = [
        "let's" , "ain't", "ya'll", "I'm", "here's", "you're",
        "that's", "he's", "she's", "it's", "we're", "they're",
        "I'll", "we'll", "you'll", "it'll", "he'll", "she'll",
        "I've", "should've", "you've", "could've", "they've",
        "I'd", "we've", "they'd", "you'd", "we'd", "he'd", "she'd",
        "didn't" "don't", "doesn't", "can't", "isn't", "aren't",
        "shouldn't", "couldn't", "wouldn't", "hasn't", "wasn't", 
        "won't", "weren't", "haven't", "hadn't"
    ]
    # count if matching word from list 
    df['contraction_count'] = df.sentence.apply(lambda x: sum(x.count(word) for word in contraction_list))

    num_cols = ['word_count',
            'char_count',
            'avg_word_length',
            'formal_pron',
            'informal_pron',
            'contraction_count']

    scalers = ["/scaler_word_count.pkl",
            "/scaler_char_count.pkl",
            "/scaler_avg_word_length.pkl",
            "/scaler_formal_pron.pkl",
            "/scaler_informal_pron.pkl",
            "/scaler_contraction_count.pkl"]

    # iterate through scaler and column names   
    # # transform numerical columns        

    for scaler, col in zip(scalers, num_cols):
        rob_scl = pickle.load(open(path+ scaler, 'rb'))
        df[col] = rob_scl.fit_transform(df[col].values.reshape(-1,1))


    return df

# load Pipelines (model + vectorizer included)
import joblib

rf_pipe = loaded_pipe = joblib.load(path+"/rf_pipe.joblib")
lr_pipe = loaded_pipe = joblib.load(path+"/lr_pipe.joblib")

# predict on test data
def predict_RF(pred_data):
    # load pipe and predict
    pred = rf_pipe.predict(pred_data)
    y_proba = rf_pipe.predict_proba(pred_data)
    y_proba = y_proba.tolist() # array to list
    #casual = 1, formal = 0
    casual = y_proba[0][1]
    formal = y_proba[0][0]

    return pred, casual, formal


def predict_LR(pred_data):
    # load pipe and predict
    pred = lr_pipe.predict(pred_data)
    y_proba = lr_pipe.predict_proba(pred_data)
    y_proba = y_proba.tolist() # array to list
    #casual = 1, formal = 0
    casual = y_proba[0][1]
    formal = y_proba[0][0]

    return pred, casual, formal


# create ensemble result
def predict(new_text):
    
    text = preprocess(new_text)
    # predict_LR
    lr_pred, lr_casual, lr_formal = predict_LR(text)
    # predict_RF 
    rf_pred, rf_casual, rf_formal = predict_RF(text)

    # taking mean of both predictions
    #pred = round((lr_pred + rf_pred)/2 , 6) # round to 6 decimal places
    casual = round((lr_casual + rf_casual)/2, 6) 
    formal = round((lr_formal + rf_formal)/2, 6) 

    # threshold of .5 
    if casual > formal:
        pred = "casual"
    if formal > casual:
        pred = "formal"
    if casual == formal:
        pred = "not available"


    return pred, casual, formal

    # return {"prediction": pred,
    #         "casual": casual,
    #         "formal": formal}



# predict function
def show_predict_page():
    st.title("Style Classifier")

    st.write("""### We need some text to predict if the sentence is casual or formal.""")
    query = st.text_area("Enter text here 👇", "", max_chars=300)
    
    if query != "":
        # run prediction function 
        pred, casual, formal = predict(query)

        st.write("Prediction: {}".format(pred))
        st.write("Casual: {}".format(casual))
        st.write("Formal: {}".format(formal))
