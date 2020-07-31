import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR


def load_df(filename):
    """Loads df if filename is valid

    :param filename:
    :return:
    """

    path = r'C:\Users\New User\PycharmProjects\engg460-research-project\Data'
    df = pd.read_pickle(os.path.join(path, filename))
    return df


def load_model(filename):
    """Load model given filename. Must be in same directory as project.

    :param filename:
    :return:
    """
    path = r'C:\Users\New User\PycharmProjects\engg460-research-project\Models'
    path = os.path.join(path, filename)
    loaded_model = pickle.load(open(path, 'rb'))

    return loaded_model


def train(data, filename=None, algo='RF'):
    """Train a model using the random forest regressor algorithm given a dataset indexed by date.
    Model is saved with Pickle, if filename is specified.
    Event occurrence values MUST be under header 'event'.

    :param algo:
    :param data:
    :param filename:
    :return:
    """
    y = data.event
    x = data.drop(['event'], axis=1)
    print(y)
    print(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    if algo == 'KNN':
        model = KNeighborsRegressor(n_neighbors=2)
        model.fit(x_train, y_train)
    elif algo == 'SVM':
        model = SVR(gamma='scale', C=1.0, epsilon=0.2)
        model.fit(x_train, y_train)
    else:
        rf = RandomForestRegressor(n_estimators=1000)
        model = rf.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print("Score:", model.score(x_test, y_test))
    print("R-Squared:", r2_score(y_test, predictions))
    df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    df1 = df.head(25)
    print(df1)
    df1.plot(kind='bar', figsize=(16, 10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    if filename is not None:
        path = r'C:\Users\New User\PycharmProjects\engg460-research-project\Models'
        path = os.path.join(path, filename)
        pickle.dump(model, open(path, 'wb'))
        model = load_model(filename)
        new_predictions = model.predict(x)
        print("New score:", model.score(x, y))
        print("New R-Squared:", r2_score(y, new_predictions))

    return


def test_model(data, filename, plot=False):
    """Test a model given dataset and model file.

    :param data:
    :param filename:
    :param plot:
    :return:
    """
    print(data)
    y = data.event
    x = data.drop(['event'], axis=1)
    model = load_model(filename)
    predictions = model.predict(x)
    print("Score:", model.score(x, y))
    df = pd.DataFrame({'Actual': y, 'Predicted': predictions})
    df = df.sort_index()
    print(df)
    if plot:
        ax = plt.gca()
        df = df.reset_index()
        df.plot(x='date', y='Actual', label='Actual values', ax=ax)
        df.plot(x='date', y="Predicted", label='Predicted values', ax=ax)
        plt.xlabel('Date')
        plt.ylabel('Event Occurrence')
        plt.legend(loc='upper right')
        plt.show()

        df.set_index('date', inplace=True)

    # feature_importances = pd.DataFrame(model.feature_importances_,
    #                                    index=x.columns,
    #                                    columns=['importance']).sort_values('importance', ascending=False)
    # print(feature_importances)
    return df
