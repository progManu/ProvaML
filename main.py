# For dealing with regular espressions
import re
# For handling natural language
import nltk
# For saving access tokens and for file management when creating and adding to the dataset
import os
# For dealing with json responses we receive from the API
import json
# For displaying the data after
import pandas as pd
# For stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# To remove stopwords
from nltk.corpus import stopwords
# To convert datetime formats
from datetime import datetime
# To exploit timezones functionalities
import pytz
# To use metrics
from sklearn import metrics
# To implement random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
# To use some utilities
import numpy as np
# To use vocabularies from natural language toolkit
from nltk.lm import Vocabulary
# To use one-hot-encoding
from sklearn.preprocessing import OneHotEncoder
# To use random forest algorithm
from sklearn.ensemble import RandomForestRegressor
# To split the dataset
from sklearn.model_selection import train_test_split
# To compare results with the dummy regressor
from sklearn.dummy import DummyRegressor


def remove_mentions_from(tweet):
    processed_text = ''
    words = tweet.split()
    for word in words:
        if not word.startswith('@'):
            processed_text = processed_text + word + ' '
    return processed_text


def remove_hashtags_from(tweet):
    processed_text = ''
    words = tweet.split()
    for word in words:
        if not word.startswith('#'):
            processed_text = processed_text + word + ' '
    return processed_text


def get_array_to_make_dict(df, field):
    array_to_make_dict = []
    for _list in df[field]:
        for tag in _list:
            array_to_make_dict.append(tag)
    return array_to_make_dict


def get_categorical_to_make_dict(df, field):
    array_to_make_dict = []
    for source in df[field]:
        array_to_make_dict.append(source)
    return array_to_make_dict


def boolean_to_numbers(df_column):

    observations = []

    for obs in df_column:
        if obs == True:
            observations.append(1)
        else:
            observations.append(0)

    return observations


if __name__ == "__main__":

    nltk.download('stopwords')

    with open('results.json') as f:
        data = json.loads(f.read())

    df = pd.json_normalize(data, record_path=['X'])

    corpus = []
    tweets = df.iloc[:, 6]

    ps = PorterStemmer()

    for tweet in tweets:

        tweet = remove_mentions_from(tweet)
        tweet = remove_hashtags_from(tweet)

        review = re.sub('[^a-zA-Z]', ' ', tweet)

        # convert all cases to lower cases
        review = review.lower()

        # split to array(default delimiter is " ")
        review = review.split()

        # loop for stemming each word
        # in string array at ith row
        review = [ps.stem(word) for word in review
                  if not word in set(stopwords.words('english'))]

        # rejoin all string array elements
        # to create back into a string
        review = ' '.join(review)
        # append each string to create
        # array of clean text
        corpus.append(review)
    df['text'] = corpus

    discriminated_sources = ['Twitter for Android',
                             'Twitter for iPhone', 'Twitter for iPad', 'Twitter Web App']

    for i, col in enumerate(df['source']):
        if col not in discriminated_sources:
            df['source'].iloc[i] = 'Other'

    for i, col in enumerate(df['user_mentions.from_tweet']):
        sum = 0
        if col is not None and len(col) != 0:
            for account in col:
                if 'followers_count' in account:  # aggiunto, probabilmente degli account privati o cancellati, basta che cerchi 1571 e vedi il succnel json per esempio
                    sum += int(account['followers_count'])

        df['user_mentions.from_tweet'].iloc[i] = sum

    for i, col in enumerate(df['user_mentions.from_retweet']):
        sum = 0
        if col is not None and len(col) != 0:
            for account in col:
                if 'followers_count' in account:
                    sum += int(account['followers_count'])

        df['user_mentions.from_retweet'].iloc[i] = sum

    current_dateTime = datetime.utcnow().replace(tzinfo=pytz.utc)
    for i, col in enumerate(df['user_created_at']):
        new_datetime = datetime.strptime(col, "%a %b %d %H:%M:%S %z %Y")
        delta_time = current_dateTime - new_datetime
        df['user_created_at'].iloc[i] = delta_time.days

    tweet_rateos = []

    for number_of_tweets, days_passed_from_creation in zip(df['tweet_count'], df['user_created_at']):
        if days_passed_from_creation != 0:
            tweet_rateos.append(number_of_tweets / days_passed_from_creation)
        else:
            tweet_rateos.append(number_of_tweets)

    df['tweet_rateo'] = tweet_rateos

    for i, col in enumerate(df['media']):
        if col is None:
            df['media'].iloc[i] = 'No Media'

    for col in df.columns:
        print(col)

    df.rename(columns={'user_mentions.from_retweet': 'retweet_followers',
              'user_mentions.from_tweet': 'tweet_followers'}, inplace=True)
    print(df.columns)

    porter = PorterStemmer()

    array_to_dict = get_array_to_make_dict(df, 'hashtags')

    hashtag_vocabulary = Vocabulary(array_to_dict, unk_cutoff=3)

    hashtags_encoding = []

    for i, _list in enumerate(df['hashtags']):
        df['hashtags'].iloc[i] = ' '.join(_list)

    hashtag_vocabulary_to_list = sorted(hashtag_vocabulary.counts)

    for string in df['hashtags']:
        encoding = np.zeros(len(hashtag_vocabulary_to_list))
        hashtags = string.split()
        for hashtag in hashtags:
            if hashtag in hashtag_vocabulary_to_list:
                index = hashtag_vocabulary_to_list.index(hashtag)
                encoding[index] = 1
        hashtags_encoding.append(encoding)

    df['hashtags'] = hashtags_encoding

    array_to_dict = get_categorical_to_make_dict(df, 'source')
    source_categorical = Vocabulary(array_to_dict, unk_cutoff=3)

    source_encoding = []

    source_categorical_to_list = sorted(source_categorical.counts)

    for source in df['source']:
        encoding = np.zeros(len(source_categorical_to_list))
        if source in source_categorical_to_list:
            index = source_categorical_to_list.index(source)
            encoding[index] = 1
        source_encoding.append(encoding)

    df['source'] = source_encoding

    array_to_dict = get_categorical_to_make_dict(df, 'text')
    text_vocabulary = Vocabulary(array_to_dict, unk_cutoff=3)

    text_encoding = []

    text_vocabulary_to_list = sorted(text_vocabulary.counts)

    for string in df['text']:
        encoding = np.zeros(len(text_vocabulary_to_list))
        words = string.split()
        for word in words:
            if word in text_vocabulary_to_list:
                index = text_vocabulary_to_list.index(word)
                encoding[index] += 1
        text_encoding.append(encoding)

    df['text'] = text_encoding

    # kf = KFold(n_splits=2, random_state = 42, shuffle = True)
    # results = []
    # # preprocessedCorpus = np.array(df) #si poteva fare solo usando numpy
    #
    # for train_index, test_index in kf.split(df):
    #     X_train, X_test = df[train_index], df[test_index]
    #     y_train, y_test = df[train_index], df[test_index]
    #
    # for train_index, test_index in kf.split(df):
    #      model = RandomForestClassifier(n_estimators = 100)
    #      model.fit(X_train, y_train)
    #      y_pred = model.predict(X_test)
    #      results.append(metrics.accuracy_score(y_test, y_pred) * 100) #se è uguale 1 altrimenti 0
    #
    # print("Accuracy: ", np.mean(results))

    array_to_dict = get_categorical_to_make_dict(df, 'media')
    media_categorical = Vocabulary(array_to_dict, unk_cutoff=1)

    media_encoding = []

    media_categorical_to_list = sorted(media_categorical.counts)

    for media in df['media']:
        encoding = np.zeros(len(media_categorical_to_list))
        if media in media_categorical_to_list:
            index = media_categorical_to_list.index(media)
            encoding[index] = 1
        media_encoding.append(encoding)
    df['media'] = media_encoding

    df['is_response'] = boolean_to_numbers(df['is_response'])
    df['is_verified'] = boolean_to_numbers(df['is_verified'])
    df['possibly_sensitive'] = boolean_to_numbers(df['possibly_sensitive'])

    df.drop(['tweet_count', 'user_created_at'], axis=1)

    y_df = pd.json_normalize(data, record_path=['Y'])
    # display(y_df)

    print(df.columns)

    print(df['hashtags'][0].shape)

    df_to_numpy = df.to_numpy()

    numpy_array = np.zeros((df_to_numpy.shape[0], 1))

    for columns in range(df_to_numpy.shape[1]):
        # print(type(df_to_numpy[0, columns]) if type(df_to_numpy[0, columns]) != type(np.zeros(1)) else df_to_numpy[0, columns].shape)
        if type(df_to_numpy[0, columns]) == type(np.zeros(1)):
            print(df_to_numpy[:, columns][0].shape)
            numpy_array = np.concatenate((numpy_array, np.zeros(
                (df_to_numpy[:, columns].shape[0], df_to_numpy[:, columns][0].shape[0]))), axis=1)
            for row_in in range(df_to_numpy[:, columns].shape[0]):
                for col_in in range(df_to_numpy[:, columns][row_in].shape[0]):
                    numpy_array[row_in][col_in +
                                        1] = df_to_numpy[:, columns][row_in][col_in]
        else:
            numpy_array = np.concatenate((numpy_array, df_to_numpy[:, columns].reshape(
                (df_to_numpy[:, columns].shape[0], 1))), axis=1)

    print(numpy_array.shape)

    numpy_array = np.delete(numpy_array, 0, axis=1)

    print(numpy_array.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        numpy_array, y_df['retweet_count'].to_numpy(), test_size=0.333, random_state=42, shuffle=True)
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
        numpy_array, y_df['like_count'].to_numpy(), test_size=0.333, random_state=1, shuffle=True)

    clf = RandomForestRegressor()
    fitted_model = clf.fit(X_train, y_train)
    fitted_model_2 = clf.fit(X_train_2, y_train_2)

    print('Accuracy: ' + str(fitted_model.score(X_test, y_test)))
    print('Accuracy_2: ' + str(fitted_model_2.score(X_test_2, y_test_2)))

    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    kf_2 = KFold(n_splits=5, random_state=1, shuffle=True)

    clf = RandomForestRegressor()

    cv_results = cross_validate(clf, numpy_array, y_df['retweet_count'].to_numpy(
    ), cv=kf, return_train_score=True, n_jobs=-1)
    cv_results_2 = cross_validate(clf, numpy_array, y_df['like_count'].to_numpy(
    ), cv=kf_2, return_train_score=True, n_jobs=-1)

    print(cv_results['test_score'].mean())
    print(cv_results['train_score'].mean())

    print(cv_results_2['test_score'].mean())
    print(cv_results_2['train_score'].mean())

    dummy_regr = DummyRegressor(strategy="mean")
    dummy_regr2 = DummyRegressor(strategy="mean")

    cv_results_dummy = cross_validate(dummy_regr, numpy_array, y_df['retweet_count'].to_numpy(
    ), cv=kf, return_train_score=True, n_jobs=-1)
    cv_results_dummy_2 = cross_validate(dummy_regr2, numpy_array, y_df['like_count'].to_numpy(
    ), cv=kf_2, return_train_score=True, n_jobs=-1)

    print(cv_results_dummy['test_score'].mean())
    print(cv_results_dummy['train_score'].mean())

    print(cv_results_dummy_2['test_score'].mean())
    print(cv_results_dummy_2['train_score'].mean())
