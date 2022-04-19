<!-- # interviewSampleCode -->
# Sample Code for OptumLabs interview

The purpose of the `ear_if.py` is to process a csv file that contains Eye-Aspect-Ratio values as a time series and apply the IsolationForest method in scikit-learn to detect outliers. Eye-Aspect-Ratio values numerically represent how open (large values) or closed (small values) the eyes of the subject were. 

The script is verified to run in **Python 3.9.7**, along with the libraries listed in the accompanying `requirements.txt` file. 

To install the requirements: 

```
pip install -r requirements.txt
```

To run the script:

```
python ear_if.py -f ear_static.csv
```

You should expect to see a csv file created, called `if.csv`, which lists whether each record is labeled as an outlier or not. 