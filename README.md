# Disaster Response Pipeline Project

- [Table of Contents](#Table_of_Contents)
  - [Introduction](#Introduction)
  - [Installation](#installation)
  - [File Descriptions](#file-descriptions)
  - [Instructions](#Instructions)
  - [Screenshots](#Screenshots)
  - [Acknowledgements](#Acknowledgements)

## Introduction
During disaster events, sending messages to appropriate disaster relief agencies on a timely manner is critical. Using natural language processing and machine learning, I built a model for an API that classifies disaster messages and also a webapp for emergency works.

- First, I developed an ETL pipeline that can:
	* Loads the messages and categories datasets
	* Merges the two datasets
	* Cleans the data
	* Stores it in a SQLite database

- Then, I created a machine learning pipeline that can:
	* Loads data from the SQLite database
	* Splits the dataset into training and test sets
	* Builds a text processing and machine learning pipeline
	* Trains and tunes a model using GridSearchCV
	* Outputs results on the test set
	* Exports the final model as a pickle file

- Finally, A **[WEB APP](https://dj-disaster-response-webapp.herokuapp.com/)** where an emergency worker can input a new message and get classification results in several categories was developed. The web app  also displays visualizations of the data.

## Installation

The code was developed using the Anaconda distribution of Python, versions 3.8.1. Python libraries used are `numpy`, `pandas`, `sqlalchemy`, `plotly`, `sklearn`, `nltk`, `pickle`, `utility`, `flask`, `wordcloud`


## File Descriptions

In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events.

* `app`
  * `templates`
    * `master.html` - main page of web app
    * `go.html` - classification result page of web app
  * `utility.py` - customized transformers and functions
  * `run.py` - Flask file that runs app

* `data`
  * `disaster_categories.csv` - data to process
  * `disaster_messages.csv` - data to process
  * `ETL_Pipeline_Preparation.ipynb` - notebook to explore the datasets and prepare ETL pipeline
  * `process_data.py` - ETL pipeline to clean and store data into a SQLite database
  * `DisasterResponse.db` - database to save clean data to

* `models`
  * `train_classifier.py` - Use ML pipeline to train and save the trained model in a pickle file
  * `classifier.pkl` - saved model
  * `ML_Pipeline_Preparation.ipynb` - notebook to try ML models and prepare ML pipeline
  * `utility.py` - customized transformers and functions

* `README.md`

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database <br>
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves <br>
      `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app. <br>
    `python run.py`

3. Go to **http://0.0.0.0:3001/**

4. The webapp files are in the **[webapp branch](https://github.com/ustcdj/DisasterResponse/tree/webapp)** <br>
The web app is at **[webapp homepage](https://dj-disaster-response-webapp.herokuapp.com/)**


## Screenshots
### 1. Home Page
<img src="images/1.jpg" width=800>

### 2. Message Categories
<img src="images/2.jpg" width=800>

## Acknowledgements

Special thanks to [Figure Eight](https://www.figure-eight.com/) for providing the dataset.
