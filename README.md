# ThoucentricProject
Tweet Sentiment Analysis

Note - Github doesn't allow uploading files > 100MB. Please download the csv file from https://drive.google.com/file/d/1RO9kKhmcxDZwG232my57fu9sHVv1btBi/view and place it in data folder inside project.

Requirements - Python 3.4+

#Steps to run this project
1. Open Git Shell and change directory where you want to download the project. ( Command - cd path/to/your/folder)
2. Clone the Git Repository to that location ( Command - git clone https://github.com/thevikingblood/ThoucentricProject.git )
3. Change directory to ThoucentricProject. ( Command - cd ThoucentricProject)
4. Install dependencies for this project in requirements.txt using pip command ( Command - pip install requirements.txt)
5. Place the "training.1600000.processed.noemoticon.csv" file inside data folder
5. Now, you can run the file in 2 ways:
  a. Using Python command line using the command - python twitteranalysis.py
  b. Using Jupytr Notebook - Open folder in Jupytr Notebook and execute twitteranalysis.ipynb
6. During execution, program will ask for keyword to search on twitter to conduct sentiment analysis ('Enter keyword you want to search on Twitter:').
7. Once executed, Word Cloud Graph will be displayed for Positive and Negative Sentiment Words.
