# CODE AUTHORED BY: ASHEN FERNANDO
# IMPLEMENT ISOLATION FOREST METHOD ON EYE-ASPECT RATIO (EAR) VALUES TO DETERMINE WHETHER A BLINK HAS OCCURED

# import the necessary packages
import pandas as pd
import argparse
from sklearn.ensemble import IsolationForest
import more_itertools as mit
import numpy as np

def findOutliers(file):

    data = pd.read_csv(file)

    # number of standard deviations away from the rolling mean 
    devs = 3

    # size of the rolling window
    roll_window = 100

    # duration to be classified as a blink, in multiples of 20, eg. dur=2 means >=60ms or ([2+1]*20) ms. Time resolution is 20 ms
    dur = 2

    # pot_outliers will contain points below 3 sigma away from rolling EAR_Avg 
    rolling = data['EAR_Avg'].rolling(roll_window).mean()
    rolling_std = rolling - devs*rolling.std()

    pot_outliers = data.loc[data['EAR_Avg'] < rolling_std]

    # a first order estimation of contamination, a ratio of data 3 sigma away from mean to total data
    contam = len(pot_outliers)/len(data)

    # implement isolation forest
    data_np = data['EAR_Avg'].to_numpy().reshape(-1,1)

    model = IsolationForest(n_estimators=100, max_samples='auto', contamination=contam, random_state=42)

    fit = model.fit(data_np)
    decision = model.decision_function(data_np)
    pred = model.predict(data_np)

    # separate outliers (with a score of -1) from normal samples

    isf = pd.DataFrame({'dec':decision, 'pred':pred})

    ears = pd.DataFrame({'inds':isf.loc[isf['pred'] == -1].index, 'EAR_vals':data['EAR_Avg'][isf.loc[isf['pred'] == -1].index]})
    ears = ears[ears['EAR_vals'] < ears['EAR_vals'].mean()]

    # creates a list of lists that keeps track of groups of consecutive records
    blinks_list_iso = [list(group) for group in mit.consecutive_groups(ears.index)]

    # counts the number of blinks and where they occur, given there are consecutive records (i.e. duration of the predicted blink) 
    # is longer than metric specified by dur
    count = 0
    blinks_iso_grouped = []
    
    for i in blinks_list_iso:
        if len(i) > dur:
            blinks_iso_grouped.append(i)
            count += 1
    
    # flatten the grouped list, to be used for validation 
    flat_list = [item for sublist in blinks_iso_grouped for item in sublist]

    # return a dataframe/csv with with 'Frame', 'EAR_Avg', 'Classification'
    data_dict = {'Frame': np.arange(0,len(data)), 'EAR_Avg': data['EAR_Avg'], 'Classification': np.zeros(len(data), dtype='int')}

    data_df = pd.DataFrame.from_dict(data_dict)

    # index into df using flat list (which has correct blink flags) to set classification value to true
    data_df['Classification'].loc[data_df.index[flat_list]] = 1

    data_df.to_csv("if.csv", index=False)

if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=True,
        help="Path to csv file with EAR values")
    args = vars(ap.parse_args())
    # print(args['file'])
    findOutliers(args['file'])







    








