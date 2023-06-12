import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3
# download necessary NLTK data
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
from sklearn.model_selection import GridSearchCV

nltk.download(['punkt', 'wordnet'])

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("DisasterResponse",'sqlite:///{}'.format(database_filepath))
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names=Y.columns
    return X, Y, category_names

def tokenize(text):
    # tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words=set(stopwords.words('english'))
    tokens = [tok for tok in tokens if tok not in stop_words]
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    # Define the model
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
    'clf__estimator__n_estimators': [10, 20],
    'clf__estimator__max_depth':[5,10],
    'clf__estimator__min_samples_split': [2, 4],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1)
    return(cv)


def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data
    y_pred = model.predict(X_test)
    # iterate over columns and print classification report for each
    for i, col in enumerate(Y_test.columns):
        print(f"Category: {col}")
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as model_filepath:
        pickle.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()