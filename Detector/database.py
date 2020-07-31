import time
from datetime import timedelta, datetime
import pandas as pd
import pymongo
from pandas.plotting import register_matplotlib_converters

from Detector.articleProcessing import (read_csv, process_search, tfidf_func)

register_matplotlib_converters()


def store_csv_data(filename):
    """ Input filename, read and process csv and then store into MongoDB database

    :param filename:
    :return:
    """
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client['perilAUS']
    col = db['events']
    data = read_csv(filename)
    col.insert_many(data)


def extract_and_insert(trove_key, search_terms, start_year, end_year, database_name, collection_name):
    """" Extract information from the Trove API, process it, and insert it into the MongoDB database

    :param collection_name: name of collection information is being stored to
    :param database_name: name of database information is being stored to
    :param end_year:
    :param start_year:
    :param search_terms:
    :param trove_key

    :return:
    """
    i = 0
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client[database_name]
    col = db[collection_name]

    skipped_years = []

    while i <= (int(end_year) - int(start_year)):
        year = int(start_year) + i

        print('Collecting articles from year:', year)
        art_col, skipped = process_search(trove_key, 'newspaper', 'Article&bulkHarvest=true', search_terms, year,
                                          year, '*',
                                          100)
        print(len(art_col), 'items inserted into database:', database_name, 'and into collection:', collection_name)

        inserted_list = art_col
        col.insert_many(inserted_list)

        if skipped:
            skipped_years.append(year)
        i += 1

    print('\t'.join(map(str, skipped_years)))
    return


def get_data_between_dates(begin_date, end_date, database_name, collection_name):
    """" Get data between specified dates from MongoDB

    :param collection_name: name of collection information is being stored to
    :param database_name: name of database information is being stored to
    :param begin_date:
    :param end_date:

    :return:
    """
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client[database_name]
    col = db[collection_name]
    if isinstance(begin_date, str):
        begin_date = datetime.strptime(begin_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    data = []
    query = col.find({'date': {'$lt': end_date, '$gt': begin_date}})

    for item in query:
        data.append(item)

    return data


def delete_data_between_dates(begin_date, end_date, database_name, collection_name):
    """ Delete all records in specified date range

    :param begin_date:
    :param end_date:
    :param database_name:
    :param collection_name:
    :return:
    """
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client[database_name]
    col = db[collection_name]

    if isinstance(begin_date, str):
        begin_date = datetime.strptime(begin_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    query = col.delete_many({'date': {'$lt': end_date, '$gt': begin_date}})

    print(query.deleted_count, "docs deleted")

    return


def process_data(data):
    """" Input tokenized textual data and output tfidf scores

    :param data:
    :return:
    """
    processed_text = []
    for article in data:
        processed_text.append(article['processed text'])

    if len(data) == 0:
        print(processed_text)
    return tfidf_func(processed_text, len(data))


def get_tfidf_in_proximity(begin_date, end_date, database_name, collection_name, proximity):
    """ Collates tf-idf values over a specified range, in chunks of size relative to proximity value. Returns a list of dataframes containing average tf-idf values of words.

    :param begin_date:
    :param end_date:
    :param database_name:
    :param collection_name:
    :param proximity:
    :return:
    """

    func_start = time.time()
    if isinstance(begin_date, str):
        begin_date = datetime.strptime(begin_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    delta = end_date - begin_date

    i = 0
    data = []
    while i < delta.days + proximity:
        from_date = begin_date + timedelta(days=i)
        to_date = from_date + timedelta(days=proximity)
        data_to_be_processed = get_data_between_dates(from_date, to_date, database_name, collection_name)
        if data_to_be_processed:
            df = process_data(data_to_be_processed)
            df['avg'] = df.mean(axis=1)
            df = df.avg
            data.append(df)
        else:
            data.append(None)
        i += 1

    func_end = time.time()
    print('Seconds elapsed during tfidf value collection:', func_end - func_start)

    return data


def get_frequency_per_day(begin_date, end_date, database_name, collection_name, proximity_overflow):
    """" Get information from MongoDB database and gather frequency of articles occurrence per day

    :param proximity_overflow:
    :param collection_name: name of collection information is being stored to
    :param database_name: name of database information is being stored to
    :param begin_date:
    :param end_date:

    :return:
    """
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client[database_name]
    col = db[collection_name]

    if isinstance(begin_date, str):
        begin_date = datetime.strptime(begin_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    delta = end_date - begin_date
    i = 0
    data = []

    while i < delta.days + proximity_overflow:
        check_date = begin_date + timedelta(days=i)
        count = col.count_documents({"date": check_date})

        date_data = [check_date, count]
        data.append(date_data)
        i += 1

    return data


def get_word_frequency_per_day(begin_date, end_date, database_name, collection_name, proximity_overflow):
    """ Similar to getting article frequency per day, but grabs from the already stored 'word frequency' and 'term frequency' columns, which hold word frequency, and word frequency over article length.
    Seen words are stored in a list, and if the word is seen again, instead of appending another value to the list, it adds to the already existing value.

    :param begin_date:
    :param end_date:
    :param database_name:
    :param collection_name:
    :param proximity_overflow:
    :return:
    """
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client[database_name]
    col = db[collection_name]

    func_start = time.time()

    if isinstance(begin_date, str):
        begin_date = datetime.strptime(begin_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    delta = end_date - begin_date
    i = 0
    data = []

    while i < delta.days + proximity_overflow:
        word_seen = []
        frequency_dict = {}
        frequency_over_article_length = {}
        check_date = begin_date + timedelta(days=i)
        query = col.find({"date": check_date})
        for article in query:
            j = 0
            for word in article['word frequency']:
                if word in word_seen:
                    frequency_dict[word] += article['word frequency'].get(word, "")
                    frequency_over_article_length[word] += article['term frequency'][j][1]
                else:
                    word_seen.append(word)
                    frequency_dict[word] = article['word frequency'].get(word, "")
                    frequency_over_article_length[word] = article['term frequency'][j][1]

                j += 1

        date_data = [check_date, frequency_dict, frequency_over_article_length]
        data.append(date_data)
        i += 1
        # print('day:', i)

    func_end = time.time()
    print('Seconds elapsed during word frequency collection:', func_end - func_start)
    return data


def get_rankings_per_day(begin_date, end_date, database_name, collection_name, proximity_overflow):

    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client[database_name]
    col = db[collection_name]
    if isinstance(begin_date, str):
        begin_date = datetime.strptime(begin_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    delta = end_date - begin_date
    i = 0
    data = []

    while i < delta.days + proximity_overflow:
        check_date = begin_date + timedelta(days=i)
        query = col.find({"date": check_date})
        one_star, two_star, three_star, four_star, five_star = 0, 0, 0, 0, 0
        for article in query:
            rating = article['auto_rating']
            if rating == 1:
                one_star += 1
            elif rating == 2:
                two_star += 1
            elif rating == 3:
                three_star += 1
            elif rating == 4:
                four_star += 1
            elif rating == 5:
                five_star += 1
        rating_dict = {
            'one': one_star,
            'two': two_star,
            'three': three_star,
            'four': four_star,
            'five': five_star
        }
        data.append(rating_dict)
        i += 1

    return data


def get_peril_dates(peril_name, begin_date, end_date, database_name, collection_name):
    """ Grab data from MongoDB using previously created function then put dates of event occurrences in a list

    :param peril_name:
    :param begin_date:
    :param end_date:
    :param database_name:
    :param collection_name:
    :return:
    """

    peril_data = get_data_between_dates(begin_date, end_date, database_name, collection_name)
    peril_list = []
    for item in peril_data:
        if item['peril'] == peril_name:
            peril_date = item['date']
            peril_list.append(peril_date)

    return peril_list


def create_ranker_df(begin_date, end_date, database_name, collection_name):
    """Create the dataframe input for LucyBot's ranker function

    :param begin_date:
    :param end_date:
    :param database_name:
    :param collection_name:
    :return:
    """

    data = get_data_between_dates(begin_date, end_date, database_name, collection_name)
    df_list = []
    i = 0
    for item in data:
        if not item.get('processed text'):
            print('empty list found, skipping')
        if len(item.get('processed text')) < 5:
            print('list less than five found')
        else:
            tokenized_text = item.get('processed text')
            joined_text = ' '.join(tokenized_text)
            item['id'] = item.get('id')
            item['articleText'] = joined_text
            df_list.append(item)
        i += 1

    df = pd.DataFrame(df_list)
    return df


def filter_and_insert(begin_date, end_date, database_name, collection_name, insert_database, insert_collection):
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client[insert_database]
    col = db[insert_collection]
    from play_ground import filter_articles

    df = filter_articles(begin_date, end_date, database_name, collection_name)

    article_list = df.to_dict('records')
    col.insert_many(article_list)

    return


