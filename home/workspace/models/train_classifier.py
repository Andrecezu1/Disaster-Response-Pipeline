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
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping

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
    '''
    load data
    Functon that loads the data from a sqlite database
    Input:
    database_filepath filepath where the database can be found
    Returns:
    X  database with the messages
    Y database with the category classigication per message
    category_names list of category names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("DisasterResponse",'sqlite:///{}'.format(database_filepath))
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names=Y.columns
    return X, Y, category_names

def tokenize(text):
    '''
    tokenize
    Functon that tokenize the text given
    Input:
    text string to be tokenized
    Returns:
    clean_tokens a list of cleaned tokens per text given
    '''
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
    '''
    Build model
    Functon that build the model used to classify the tokens
    Returns:
    cv a model tunned with grid search
    '''
    # Define the model
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = [{
        'clf__estimator__n_estimators': [100, 300],
        'clf__estimator__max_depth': [5, 10],
        'clf__estimator__min_samples_split': [2, 5]
    }]
    cv = GridSearchCV(pipeline, param_grid=parameters,cv=3, scoring='accuracy',n_jobs=-1)
    return(pipeline)


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model
    Function that print the result of the model evaluated in the test set
    Input:
    model the trained model
    X_test the set of messages used to test the model
    Y_test the classification of the test messages
    category_names the list of category names
    '''
    # predict on test data
    y_pred = model.predict(X_test)
    # iterate over columns and print classification report for each
    for i, col in enumerate(Y_test.columns):
        print(f"Category: {col}")
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    '''
    save model
    Function that saves the trained model
    Input:
    model the trained model
    model_filepath the filepath where the model is going to be saved
    Output:
    saved model
    '''
    with open(model_filepath, 'wb') as model_filepath:
        pickle.dump(model, model_filepath)


def main():
    '''
    main
    Main function that process the functions to return the model.
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        
        model = build_model()
        
        print('Training model...')
        # fit model
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