# Disaster-Response-Pipeline
### Table of Contents

1. [Installation](#installation)
2. [Execution instructiones](#execution)
3. [Project Motivation](#motivation)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## Execution instructions <a name="execution"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## Project Motivation<a name="motivation"></a>

When a disaster occurs, there is a need of instantanious information in order to deploy the emergency response forces. In this project I analyze the messages shared during a disaster, build a text processing pipeline to classify the messages delivered when a dissaster occurs.

The data was given by Udacity and consist in four files:
1. categories.csv
2. DisasterResponseDB
3. messages.csv
4. Twitter-sentiment-self-drive-DFE.csv

## File Descriptions <a name="files"></a>

There are two (2) notebooks and one zip file:
1. "ETL Pipeline Preparation.ipynb" is the ETL pipeline used to prepare the data.
2. "ML Pipeline Preparation.ipynb" is the pipeline that trains a classifier to analyze the data.
3. In the home folder you can find all the files you need to run the project. 
3.1 disaster_categories.csv File with the categories of the messages
3.2 disaster_messages.csv file with the messages
3.3 process_data.py code to load and transform the data that set the clean data in a database.
3.4 train_classifier.py code that set the model which will classify the text of the messages.
3.5 run.py python code that create the web page
3.6 go.htlm html code used to classify the message entered in the page
3.7 master.html code used to create the web page.
5. "Disasters.html"  is the web page where the search can be done.

## Results<a name="results"></a>

The main findings of the code can be found at the web page found in the Disasters.html file.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to UDACITY for the data and templates.  Feel free to use the code here as you would like! 

