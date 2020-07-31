import time
from datetime import timedelta
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from statsmodels.formula.api import ols

from Detector.database import (get_data_between_dates, get_peril_dates, process_data, get_word_frequency_per_day,
                               get_frequency_per_day, get_tfidf_in_proximity, get_rankings_per_day)


def compile_values(peril_data, article_frequency_data, term_frequency_data, proximity_range, keyword, weighted_values):
    """ OLD function, no longer in use due to little modularisation and flexibility.
    Grab values from article freq data and word freq data, and put them together, such that accessing them and inputting them into a Data frame is easier.

    :param peril_data:
    :param article_frequency_data:
    :param term_frequency_data:
    :param proximity_range:
    :param keyword:
    :param weighted_values:
    :return:
    """
    day_weight = 1
    event_occurrence = False
    for i in range(0, (len(article_frequency_data) - proximity_range)):
        proximity = 0
        try:
            article_frequency_data[i].append(float(int(term_frequency_data[i][1].get(keyword, ""))))
            article_frequency_data[i].append(term_frequency_data[i][2].get(keyword, ""))
        except (IndexError, ValueError):
            article_frequency_data[i].append(0)
            article_frequency_data[i].append(0)
        while proximity < proximity_range:
            weight = 1
            if weighted_values:
                weight = 1 - ((proximity_range - proximity) / proximity_range * 0.1)
            article_frequency_data[i][1] += (article_frequency_data[i + proximity + 1][1] * weight)

            try:
                article_frequency_data[i][2] += float(int(term_frequency_data[i + proximity + 1][1].get(keyword, "")))
                article_frequency_data[i][3] += term_frequency_data[i + proximity + 1][2].get(keyword, "")
            except (IndexError, ValueError):
                article_frequency_data[i][2] += 0
                article_frequency_data[i][3] += 0
            proximity += 1

        if day_weight == 30:
            event_occurrence = False
        if article_frequency_data[i][0] in peril_data:
            event_occurrence = True
            day_weight = 1
            article_frequency_data[i].append(0)
        elif event_occurrence:
            article_frequency_data[i].append(day_weight)
            day_weight += 1
        else:
            article_frequency_data[i].append(100)

    del article_frequency_data[(len(article_frequency_data) - proximity_range):]
    print(article_frequency_data)
    return article_frequency_data


def correlate_frequency_event_occurrence(peril_name, proximity_range, begin_date, end_date, peril_database,
                                         peril_collection, article_database, article_collection, keyword,
                                         weighted_values=True, plot=False):
    """ OLD function, no longer in use due to little modularisation and flexibility.
    Get frequency per day and event occurrence, put them into a DataFrame and correlate. Also produce a plot showing both.

    :param plot:
    :param keyword:
    :param weighted_values:
    :param peril_name:
    :param proximity_range:
    :param begin_date:
    :param end_date:
    :param peril_database:
    :param peril_collection:
    :param article_database:
    :param article_collection:
    :return:
    """
    peril_data = get_peril_dates(peril_name, begin_date, end_date, peril_database, peril_collection)
    article_data = get_frequency_per_day(begin_date, end_date, article_database, article_collection, proximity_range)
    word_frequency_data = get_word_frequency_per_day(begin_date, end_date, article_database, article_collection,
                                                     proximity_range)
    data = compile_values(peril_data, article_data, word_frequency_data, proximity_range, keyword, weighted_values)

    func_start = time.time()
    df = pd.DataFrame(data)
    scaler = preprocessing.MinMaxScaler()
    print(df)
    df = df[df[4] != 100]
    print(df)
    df[[1, 2, 3, 4]] = scaler.fit_transform(df[[1, 2, 3, 4]])
    print(df)

    correlation_val = df[1].corr(df[4])
    correlation_val2 = df[2].corr(df[4])
    correlation_val3 = df[3].corr(df[4])
    func_end = time.time()
    print('Seconds elapsed during standardization:', func_end - func_start)

    df.columns = ['date', 'art_freq', 'word_freq', 'word_len_freq', 'event']
    print(df.corr())
    model = ols("event ~ art_freq", data=df).fit()
    print(model.params)
    print(model.summary())

    if plot:
        ax = plt.gca()
        df.plot(x=0, y=1, label='Article Frequency in Range', ax=ax)
        df.plot(x=0, y=2, label='Word Frequency in Range', ax=ax)
        df.plot(x=0, y=3, label='Word Frequency over Article Length in Range', ax=ax)
        df.plot(x=0, y=4, label='Event Occurrence', ax=ax)
        plt.xlabel('Date')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.show()

    print(correlation_val, 'with a proximity range of', proximity_range)
    print(correlation_val2, 'with a proximity range of', proximity_range)
    print(correlation_val3, 'with a proximity range of', proximity_range)

    # change date from column value to index for later training and testing
    df.set_index('date', inplace=True)

    return df


def gather_training_data(peril_name, proximity_range, begin_date, end_date, peril_database,
                         peril_collection, article_database, article_collection):
    """Gathers data from MongoDB, and inserts them into one single list so that accessing is easier

    :param peril_name:
    :param proximity_range:
    :param begin_date:
    :param end_date:
    :param peril_database:
    :param peril_collection:
    :param article_database:
    :param article_collection:
    :return:
    """
    peril_data = get_peril_dates(peril_name, begin_date, end_date, peril_database, peril_collection)

    data = []
    print(len(peril_data), 'perils found')

    for item in peril_data:
        peril_start = item
        peril_end = item + timedelta(days=30)
        article_data = get_frequency_per_day(peril_start, peril_end, article_database, article_collection,
                                             proximity_range)
        word_frequency_data = get_word_frequency_per_day(peril_start, peril_end, article_database, article_collection,
                                                         proximity_range)
        tfidf_data = get_tfidf_in_proximity(peril_start, peril_end, article_database, article_collection,
                                            proximity_range)
        ranking_data = get_rankings_per_day(peril_start, peril_end, article_database, article_collection,
                                            proximity_range)
        i = 0
        for element in article_data:
            try:
                freq = {
                    'date': element[0],
                    'article_freq': element[1],
                    'word_freq': word_frequency_data[i][2],
                    'tfidf_val': tfidf_data[i],
                    'ranks': ranking_data[i],
                    'event_value': i
                }
                data.append(freq)
            except IndexError:
                print(element[1], 'articles found on', element[0], 'inserting a None row')
                freq = {
                    'date': element[0],
                    'article_freq': element[1],
                    'word_freq': None,
                    'tfidf_val': None,
                    'ranks': ranking_data[i],
                    'event_value': i
                }
                data.append(freq)
            i += 1

    return data


def gather_all_data(peril_name, proximity_range, begin_date, end_date, peril_database,
                    peril_collection, article_database, article_collection):
    """Gathers ALL data from within specified time range. (As opposed to how the other only grabs articles within range of an event)

    :param peril_name:
    :param proximity_range:
    :param begin_date:
    :param end_date:
    :param peril_database:
    :param peril_collection:
    :param article_database:
    :param article_collection:
    :return:
    """
    data = []

    article_data = get_frequency_per_day(begin_date, end_date, article_database, article_collection,
                                         proximity_range)
    word_frequency_data = get_word_frequency_per_day(begin_date, end_date, article_database, article_collection,
                                                     proximity_range)
    tfidf_data = get_tfidf_in_proximity(begin_date, end_date, article_database, article_collection,
                                        proximity_range)
    peril_data = get_peril_dates(peril_name, begin_date, end_date, peril_database, peril_collection)
    ranking_data = get_rankings_per_day(begin_date, end_date, article_database, article_collection,
                                        proximity_range)

    i = 0
    day_value = 0
    none_added = 0
    event_occurrence = False

    for element in article_data:
        if day_value == 30:
            event_occurrence = False
        if element[0] in peril_data:
            event_occurrence = True
            day_value = 0
        elif event_occurrence:
            day_value += 1
        else:
            day_value = 100
        try:
            tfidf = tfidf_data[i]
        except IndexError as e:
            print('tfidf missing val', e)
        try:
            freq = {
                'date': element[0],
                'article_freq': element[1],
                'word_freq': word_frequency_data[i][2],
                'tfidf_val': tfidf_data[i],
                'ranks': ranking_data[i],
                'event_value': day_value
            }
            data.append(freq)
        except IndexError:
            print(element[1], 'articles found on', element[0], 'inserting a None row')
            freq = {
                'date': element[0],
                'article_freq': element[1],
                'word_freq': word_frequency_data[i][2],
                'tfidf_val': tfidf_data[i],
                'ranks': ranking_data[i],
                'event_value': day_value
            }
            data.append(freq)
            none_added += 1
        i += 1
    print('None rows added:', none_added)

    return data


def build_dataset(peril_name, proximity_range, begin_date, end_date, peril_database,
                  peril_collection, article_database, article_collection, keyword_list, except_keywords,
                  weighted_values=True, plot=False, filename=None, train=False):
    """This is the main function that builds our dataset. It returns our dependent variable and features in a single dataframe indexed by date.
    Also returns correlation values of features and plots data if specified to. Can also optionally plot, and save the final dataframe.

    :param train:
    :param filename:
    :param peril_name:
    :param proximity_range:
    :param begin_date:
    :param end_date:
    :param peril_database:
    :param peril_collection:
    :param article_database:
    :param article_collection:
    :param keyword_list:
    :param weighted_values:
    :param plot:
    :return:
    """
    if train is True:
        data = gather_training_data(peril_name, proximity_range, begin_date, end_date, peril_database, peril_collection,
                                    article_database, article_collection)
        print('data length is', len(data))
    else:
        data = gather_all_data(peril_name, proximity_range, begin_date, end_date, peril_database, peril_collection,
                               article_database, article_collection)
        print('data length is', len(data))

    date_list = []
    event_list = []
    article_frequency_list = create_feature_article_frequency(data, proximity_range, weighted_values)

    for item in data:
        date_list.append(item['date'])
        event_list.append(item['event_value'])

    df = pd.DataFrame({'date': date_list})
    df['event'] = event_list

    df.drop(df.tail(proximity_range).index, inplace=True)
    df['article frequency'] = article_frequency_list

    for keyword in keyword_list:
        header = ('"' + keyword + '"')
        if keyword in except_keywords:
            word_frequency_list = create_feature_word_freq(data, proximity_range, keyword, weighted_values)
            df[header + ' frequency'] = word_frequency_list
        tfidf_list = create_feature_word_tfidf(data, proximity_range, keyword, weighted_values)
        df[header + ' tfidf value'] = tfidf_list

    one_list, two_list, three_list, four_list, five_list, average_list = create_feature_rankings(data, proximity_range, weighted_values)
    # df['one star frequency'] = one_list
    # df['two star frequency'] = two_list
    # df['three star frequency'] = three_list
    # df['four star frequency'] = four_list
    df['five star frequency'] = five_list
    df['avg star frequency'] = average_list

    df = normalize_df(df, train)
    print(df.corr())

    if plot:
        plot_df(df)

    df.set_index('date', inplace=True)
    if filename is not None:
        path = r'C:\Users\New User\PycharmProjects\engg460-research-project\Data'
        path = os.path.join(path, filename)
        df.to_pickle(path)

    return df


def plot_df(df, load=False):
    """Plot the whole of the dataset, works only specific to the dataframe built in build_dataset.
    Added parameter incase we want to plot an already saved dataframe.

    :param load:
    :param df:
    :return:
    """
    if load:
        df.reset_index(level=0, inplace=True)
    column_list = list(df)
    index = column_list[0]
    column_list = column_list[1:]
    df.plot(x=index, y=column_list)
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

    return


def normalize_df(df, train=False):
    """Normalizes the dataframe with a minmax scaler, such that all values lie within range 0 - 1.
    Achieves this by first indexing date, then resetting it (as it breaks normalization).
    Additional argument for whether it is for training data or not, to remove any data outside of range

    :param train:
    :param df:
    :return:
    """
    if train:
        df = df[df['event'] <= 29]
    df.set_index('date', inplace=True)
    normalized_df = (df - df.min()) / (df.max() - df.min())
    normalized_df = normalized_df.loc[~normalized_df.index.duplicated(keep='last')]
    normalized_df.reset_index(level=0, inplace=True)
    return normalized_df


def create_feature_article_frequency(data, proximity_range, weighted_values=True):
    """ Creates a list of article frequencies corresponding to dates.

    :param data:
    :param proximity_range:
    :param weighted_values:
    :return:
    """
    article_frequency = []
    i = 0
    while i < (len(data) - proximity_range):
        proximity = 0
        article_freq_value = 0
        weight = 1

        while proximity < proximity_range:
            if weighted_values:
                weight = 1 * ((proximity_range - proximity) / proximity_range)

            article_freq_value += (data[i + proximity]['article_freq'] * weight)
            proximity += 1

        article_frequency.append(article_freq_value)
        i += 1

    return article_frequency


def create_feature_word_freq(data, proximity_range, keyword, weighted_values=True):
    """Creates a list of word frequencies according to chosen keyword corresponding to dates.

    :param data:
    :param proximity_range:
    :param keyword:
    :param weighted_values:
    :return:
    """
    word_frequency = []
    i = 0
    zeroes_added = 0
    while i < (len(data) - proximity_range):
        proximity = 0
        word_freq_value = 0
        weight = 1

        while proximity < proximity_range:
            if weighted_values:
                weight = 1 * ((proximity_range - proximity) / proximity_range)
            if data[i + proximity]['word_freq'] is None:
                freq_val = 0
                zeroes_added += 1
            else:
                freq_val = data[i + proximity]['word_freq'].get(keyword, "")
            if isinstance(freq_val, str):
                freq_val = 0
                zeroes_added += 1
            word_freq_value += (freq_val * weight)
            proximity += 1

        word_frequency.append(word_freq_value)
        i += 1
    print(i, 'items processed for word freq')
    print(zeroes_added, 'zeroes added')
    return word_frequency


def create_feature_word_tfidf(data, proximity_range, keyword, weighted_values=True):
    """Creates a list of average tf-idf values for the keyword corresponding to dates.

    :param data:
    :param proximity_range:
    :param keyword:
    :param weighted_values:
    :return:
    """
    tfidf_list = []
    i = 0
    while i < (len(data) - proximity_range):
        proximity = 0
        tfidf_val_over_range = 0
        weight = 1

        while proximity < proximity_range:
            if weighted_values:
                weight = 1 * ((proximity_range - proximity) / proximity_range)
            try:
                if data[i + proximity]['tfidf_val'] is None:
                    tfidf_val_over_range += 0
                else:
                    tfidf_val = data[i + proximity]['tfidf_val'].loc[keyword]
                    tfidf_val_over_range += (tfidf_val * weight)
                proximity += 1
            except KeyError:
                tfidf_val_over_range += 0
                proximity += 1

        tfidf_list.append(tfidf_val_over_range)
        i += 1
    return tfidf_list


def create_feature_rankings(data, proximity_range, weighted_values=True):
    """

    :param data:
    :param proximity_range:
    :param weighted_values:
    :return:
    """

    one_list = []
    two_list = []
    three_list = []
    four_list = []
    five_list = []
    average_list = []
    i = 0
    while i < (len(data) - proximity_range):
        one_star, two_star, three_star, four_star, five_star = 0, 0, 0, 0, 0
        proximity = 0
        weight = 1

        while proximity < proximity_range:
            if weighted_values:
                weight = 1 * ((proximity_range - proximity) / proximity_range)
            day_ranks = data[i + proximity]['ranks']
            one_star += day_ranks['one'] * weight
            two_star += day_ranks['two'] * weight
            three_star += day_ranks['three'] * weight
            four_star += day_ranks['four'] * weight
            five_star += day_ranks['five'] * weight

            proximity += 1

        average_rank = ((one_star * 1) + (two_star * 2) + (three_star * 3) + (four_star * 4) + (five_star * 5)) / 5
        one_list.append(one_star)
        two_list.append(two_star)
        three_list.append(three_star)
        four_list.append(four_star)
        five_list.append(five_star)
        average_list.append(average_rank)

        i += 1
    return one_list, two_list, three_list, four_list, five_list, average_list


def find_important_keywords(peril_name, proximity_range, begin_date, end_date, peril_database,
                            peril_collection, article_database, article_collection):
    """Looks at the 14 days after every peril and finds the average tf-idf values of keywords according to that corpus

    :param peril_name:
    :param proximity_range:
    :param begin_date:
    :param end_date:
    :param peril_database:
    :param peril_collection:
    :param article_database:
    :param article_collection:
    :return:
    """
    peril_data = get_peril_dates(peril_name, begin_date, end_date, peril_database, peril_collection)
    data = []
    print(len(peril_data), 'perils found')
    for item in peril_data:
        article_data = get_data_between_dates(item, item + timedelta(proximity_range), article_database,
                                              article_collection)
        # print(len(article_data), article_data[0]['date'])
        data.extend(article_data)

    df = process_data(data)
    df['avg'] = df.mean(axis=1)
    df = df.sort_values(by=['avg'], ascending=False)
    print(df)
    df = df.avg
    print(df.head(25))
    return df


def keep_columns(df, columns_keep):
    """get a dataframe and chosen columns, and keep those while dropping the rest

    :param df:
    :param columns_keep:
    :return:
    """

    columns_keep.append('event')
    column_list = list(df.columns)
    for item in columns_keep:
        if item in column_list:
            column_list.remove(item)

    for item in column_list:
        print('removing column:', item)
        df = df.drop(item, 1)

    return df
