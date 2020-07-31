import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from datetime import timedelta, datetime

from Detector.database import get_data_between_dates, get_peril_dates


def create_confidence_interval(df, threshold, filename=None):
    """Creates a confidence interval from an input dataframe.
    Takes all predicted values under a certain threshold, and outputs confidence intervals of event occurrences

    :param filename:
    :param threshold:
    :param df:
    :return:
    """
    dates = []
    for row in df.itertuples():
        predicted_val = row.Predicted
        date = row.Index

        if predicted_val < threshold:
            confidence = 1 - (predicted_val/threshold)
            date_dict = {
                'date': date,
                'confidence': confidence
            }
            dates.append(date_dict)
    confidence_df = pd.DataFrame(dates)

    confidence_df.set_index('date', inplace=True)
    print(confidence_df)

    if filename is not None:
        path = r'C:\Users\New User\PycharmProjects\engg460-research-project\Data'
        path = os.path.join(path, filename)
        confidence_df.to_pickle(path)
    return confidence_df


def group_events(df):

    group_date = None
    dates = []
    df_length = len(df.index)
    i = 0
    for row in df.itertuples():
        if group_date is None:
            group_date = row.Index
            group_confidence = row.confidence
        else:
            date = row.Index
            confidence = row.confidence
            difference = (date - group_date).days
            if difference <= 3:
                if confidence > group_confidence:
                    group_confidence = confidence
            else:
                date_dict = {
                    'date': group_date,
                    'confidence': group_confidence
                }
                dates.append(date_dict)
                group_date = row.Index
                group_confidence = row.confidence
        i += 1
        if i == df_length:
            date_dict = {
                'date': group_date,
                'confidence': group_confidence
            }
            dates.append(date_dict)
    filtered_df = pd.DataFrame(dates)
    filtered_df.set_index('date', inplace=True)

    return filtered_df


def filter_known_events(df, peril_database, peril_collection, begin_date, end_date):
    unknown_events = []
    known_events = []
    dates = []
    peril_data = get_peril_dates('Tropical Cyclone', begin_date, end_date, peril_database, peril_collection)
    peril_len = len(peril_data)
    for item in peril_data:
        print(item)

    for row in df.itertuples():
        known_event = False
        date = row.Index
        i = -3
        date_dict = {}

        while i < 10:
            date = date + timedelta(i)
            if date in peril_data:
                known_event = True
                peril_data.remove(date)
                date_dict = {
                    'date': row.Index,
                    'confidence': row.confidence,
                    'event': True
                }
            i += 1
        if known_event is False:
            date_dict = {
                'date': row.Index,
                'confidence': row.confidence,
                'event': False
            }
            unknown_events.append(row)
        else:
            known_events.append(row)
        dates.append(date_dict)

    all_events = pd.DataFrame(dates)
    all_events.set_index('date', inplace=True)
    print(all_events)
    print(len(known_events), 'perils found out of', peril_len)
    print(peril_data)

    true_pos = len(known_events)
    false_neg = peril_len-len(known_events)
    false_pos = len(all_events) - len(known_events)

    print('True Positives:', true_pos)
    print('False Negatives:', peril_len-len(known_events))
    print('False Positives:', len(all_events) - len(known_events))

    precision = true_pos/(true_pos + false_pos)
    recall = true_pos/(true_pos + false_neg)

    if true_pos == 0:
        f1_score = 0
    else:
        f1_score = 2*((precision*recall)/(precision+recall))
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score', f1_score)

    return all_events


def return_ids(df, article_database, article_collection, minimum_confidence, proximity):
    """
    
    :param proximity:
    :param df:
    :param article_database:
    :param article_collection:
    :param minimum_confidence:
    :return:
    """
    all_ids = []
    for row in df.itertuples():
        ids_in_range = []
        if row.confidence > minimum_confidence:
            begin = row.Index
            end = row.Index + timedelta(days=proximity)
            print('Begin is:', begin, 'and end is:', end)
            data = get_data_between_dates(begin, end, article_database, article_collection)
            i = 0
            for item in data:
                article_id = item.get('id')
                ids_in_range.append(article_id)
                i += 1
        if ids_in_range:
            entry = {
                'begin': begin,
                'end': end,
                'confidence': row.confidence,
                'ids': ids_in_range

            }
            all_ids.append(entry)

    all_ids = pd.DataFrame(all_ids)

    print(all_ids)
    return all_ids
