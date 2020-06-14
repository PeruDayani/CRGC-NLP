#!/usr/bin/env python3

from flask import Flask, render_template, request, jsonify
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from collections import Counter
import pandas as pd
import numpy as np
import nltk
import collections 
import re
import smtplib, ssl
import joblib
import pickle

initRun = True
crgcDataPath = "data/crgcSample.csv"

crgcData = pd.DataFrame()

signWordsDf = pd.DataFrame()
signBigramsDf = pd.DataFrame()
signTrigramsDf = pd.DataFrame()

signWords = []
signBigrams = []
signTrigrams = []

models = {}
vectorizers = {}

models_vects_cols = ['adText_Final_DX','adText_OP_Proc','adText_Path','adText_Phys_EX','adText_Remarks','adText_Scopes','adText_Surg_1']
models_path = "models/"

master_model = None

def init():

    global initRun 
    global crgcDataPath

    global crgcData

    global signWordsDf
    global signBigramsDf
    global signTrigramsDf

    global signWords
    global signBigrams 
    global signTrigrams 

    global models
    global vectorizers

    global models_vects_cols
    global models_path

    global master_model

    print("Initializing dataframes")

    crgcData = pd.read_csv(crgcDataPath)
    helper = {True: 1, False:0}
    crgcData.replace(helper, inplace=True)

    signWordsDf = pd.read_csv('data/significantWords')
    signWordsDf = signWordsDf.drop(['Unnamed: 0'],axis=1)

    signBigramsDf = pd.read_csv('data/significantBigrams')
    signBigramsDf = signBigramsDf.drop(['Unnamed: 0'],axis=1)

    signTrigramsDf = pd.read_csv('data/significantTrigrams')
    signTrigramsDf = signTrigramsDf.drop(['Unnamed: 0'],axis=1)

    signWords = signWordsDf['Word'].tolist()
    signBigrams = signBigramsDf['Word'].apply(eval).tolist()
    signTrigrams = signTrigramsDf['Word'].apply(eval).tolist()

    for col in models_vects_cols:
    
        model_name = models_path + col + "_final_model.joblib"
        vect_name = models_path + col + "_final_vectorizer.pickle"
        
        # load the model from disk
        loaded_model = joblib.load(model_name)

        # load the vectorizer from the disk
        loaded_vectorizer = pickle.load(open(vect_name, "rb"))
        
        models[col] = loaded_model
        vectorizers[col] = loaded_vectorizer    

    master_model = joblib.load('models/Master_Final_model.joblib')

    initRun = False

def classify(df):

    global master_model
    global models
    global vectorizers
    
    dfResults = pd.DataFrame(columns = models_vects_cols)
    
    for col in models_vects_cols:
        
        vectorizer = vectorizers[col]
        model = models[col]

        data = df[col]
        X = vectorizer.transform(data)
        y_pred = model.predict(X.toarray())
        
        dfResults[col] = y_pred
        
    return dfResults

def runModel(df):

    final_result = []

    df = cleanData(df)

    for i in range(df.shape[0]):

        row_df = df.iloc[[i]]

        row_classification = classify(row_df)
        row_prediction = master_model.predict(row_classification)[0]
        row_classification = row_classification.replace({1: "Muscle Present", 0: "Muscle Not Present"})

        final_result.append( (row_classification, row_prediction) )    

    return final_result

def emailAlert(receiver_email):

    port = 587  
    smtp_server = "smtp.gmail.com"
    sender_email = "crgcxucb@gmail.com"
    password = 'crgcXucb2020'
    message = """\
    Subject: CRGCxUCB

Dear Pathologist, 
            
Could you please check if sufficient muscle has been resected ? 
We do not seem to find the required information in the report. 

Thanks, 
Cancer Registry of Greater California Staff"""

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.ehlo()  # Can be omitted
        server.starttls(context=context)
        server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

def sample():
    if initRun:
        init()

    test_df = crgcData.sample() 
    return test_df

def my_tokenizer(text):
    return text.split() if text != None else []

def word_counts(df, col):
    tokens = df[col].apply(my_tokenizer).sum()
    counter = Counter(tokens)
    return list(counter.elements()) 

def bigram_counts(df, col):
    counts = collections.Counter()
    df[col] = df[col].fillna('')
    for sent in df[col]:
        words = nltk.word_tokenize(sent)
        counts.update(nltk.bigrams(words))
    return list(counts.elements())

def trigram_counts(df, col):
    counts = collections.Counter()
    df[col] = df[col].fillna('')
    for sent in df[col]:
        words = nltk.word_tokenize(sent)
        counts.update(nltk.trigrams(words))
    return list(counts.elements())


def findSignificantPhases(df):
    
    if initRun:
        init()

    final_result = []

    for i in range(df.shape[0]):

        row_df = df.iloc[[i]]

        result = pd.DataFrame(columns=['Word','Total','True','False', 'Significance', 'Classification'])
      
        words = []
        bigrams = []
        trigrams = []

        for col in row_df.columns:
            
            words += word_counts(row_df, col)
            bigrams += bigram_counts(row_df, col)
            trigrams += trigram_counts(row_df, col)

            
        for word in words:
            if word in signWords:
                #print(word)
                data = signWordsDf.loc[signWordsDf['Word'] == word]
                #print(data)
                result = result.append(data, ignore_index=True)
                #print(result)
                
                
        for bigram in bigrams:
            if bigram in signBigrams:
                #print(bigram)
                data = signBigramsDf.loc[signBigramsDf['Word'] == str(bigram)]
                #print(data)
                result = result.append(data, ignore_index=True)
                #print(result)
                
            
        for trigram in trigrams:
            if trigram in signTrigrams:
                #print(trigram)
                data = signTrigramsDf.loc[signTrigramsDf['Word'] == str(trigram)]
                #print(data)
                result = result.append(data, ignore_index=True)
                #print(result)
           

        result['Classification'] = result['Classification'].map({True: "Muscle Present", False: "Muscle Not Present"})
        result.drop_duplicates()

        final_result.append(result)

    return final_result

def remove_accented_chars(text):
    import unicodedata
    if not pd.isnull(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_stopwords(text, is_lower_case=False):
    from nltk.corpus import stopwords
    stopword = stopwords.words('english')
    
    filtered_text = text
    if not pd.isnull(text):
        tokens = nltk.word_tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopword]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopword]
        filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def remove_special_characters(text, remove_digits=True):
    import re
    if not pd.isnull(text):
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
    return text

def convert_lowercase(text):
    if not pd.isnull(text):
        text = text.lower()
    return text

def cleanData(df):
    
    crgcTextualDataColumns = ['adText_Final_DX','adText_OP_Proc','adText_Other_RX','adText_Path','adText_Phys_EX','adText_Remarks','adText_Scopes','adText_Surg_1','adText_Surg_2','adText_Surg_3']
    
    for col in df.columns:
        
        if col in crgcTextualDataColumns:
            df[col] = df[col].apply(convert_lowercase)
            df[col] = df[col].apply(remove_special_characters)
            #df[col] = df[col].apply(remove_stopwords)
            df[col] = df[col].apply(remove_accented_chars)
            df[col] = df[col].fillna('')
        else:
            #print("Unparesable column, therefore dropped : " + col)
            df = df.drop([col], axis=1)
            
    return df

def extractPathInfo(df):

    path_name_col = "path_Name"
    path_email_col = "path_Email"
    path_number_col = "path_Number"

    if path_name_col not in df.columns:
        return None
    if path_email_col not in df.columns:
        return None
    if path_number_col not in df.columns:
        return None

    name = df[path_name_col][0]
    email = df[path_email_col][0]
    number = df[path_number_col][0]

    return (name, email, number)


def toDataframe(data):
    crgcColumns = ['adText_Final_DX','adText_OP_Proc','adText_Path','adText_Phys_EX','adText_Remarks','adText_Scopes','adText_Surg_1']
    df = pd.DataFrame(np.array([data]),columns=crgcColumns)
    return df

