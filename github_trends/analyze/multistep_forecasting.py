from keras.models import Sequential
from keras.layers import *
import pandas
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

from github_trends.services.database_service import DatabaseService

class Forecasting:
    def __init__(self):
        self.db_service = DatabaseService()

    def arrange_input_sequence(self, data, n_in, n_out, input_columns, output_columns):
        if input_columns is None:
            input_columns = list(data.columns.values)  # All columns are input by default
        if output_columns is None:
            output_columns = [data.columns[-1]]  # Last column is output by default

        input_data = data[input_columns]
        output_data = data[output_columns]
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(input_data.shift(i))
            names += [(str(column_name) + '(t-%d)' % i) for column_name in input_columns]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(output_data.shift(-i))
            if i == 0:
                names += [(str(column_name) + '(t)') for column_name in output_columns]
            else:
                names += [(str(column_name) + '(t+%d)' % i) for column_name in output_columns]

        # put it all together
        agg = pandas.concat(cols, axis=1)
        agg.columns = names
        agg.dropna(inplace=True)

        return agg

    def forecast_stars_of_repo(self, repo_id, data_df, input_features, output_features):
        scaler = MinMaxScaler(feature_range=(-1, 1))

        n_input_steps = 90
        n_in_variables = len(input_features)
        n_out_variables = 7 * len(output_features)
        n_obs = n_input_steps * n_in_variables

        sequence_data = self.arrange_input_sequence(data_df, n_input_steps, n_out_variables,
                                                    input_features, output_features)

        values = sequence_data.values
        values = scaler.fit_transform(values)

        training_weeks = np.math.floor(sequence_data.shape[0] * (2 / 3))
        headers = sequence_data.columns
        indexes = sequence_data.index.values

        train = pandas.DataFrame(values[:training_weeks, :], columns=headers, index=indexes[:training_weeks])
        test = pandas.DataFrame(values[training_weeks:, :], columns=headers, index=indexes[training_weeks:])

        train_data = pandas.DataFrame(train.values[:, :n_obs], columns=train.columns.values[:n_obs], index=train.index.values)
        train_y = pandas.DataFrame(train.values[:, n_obs:], columns=train.columns.values[n_obs:], index=train.index.values)
        test_data = pandas.DataFrame(test.values[:, :n_obs], columns=test.columns.values[:n_obs], index=test.index.values)
        test_y = pandas.DataFrame(test.values[:, n_obs:], columns=test.columns.values[n_obs:], index=test.index.values)

        train_X = train_data.values.reshape((train_data.values.shape[0], n_input_steps, n_in_variables))
        test_X = test_data.values.reshape((test_data.values.shape[0], n_input_steps, n_in_variables))

        model = Sequential()
        lstm = LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]))
        rnn = SimpleRNN(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True)
        model.add(rnn)
        model.add(lstm)
        model.add(Dense(n_out_variables))
        model.compile(loss='mse', optimizer='adam')

        history = model.fit(train_X, train_y, epochs=50, batch_size=32, verbose=1, validation_split=0.05, shuffle=False)
        pred = model.predict(test_X, verbose=1)

        test_X = test_X.reshape((test_X.shape[0], n_obs))
        inv_pred = np.concatenate((test_X, pred), axis=1)
        inv_pred = scaler.inverse_transform(inv_pred)
        inv_pred = inv_pred[:, -n_out_variables:]
        inv_pred2 = inv_pred[:,-n_out_variables]

        inv_y = np.concatenate((test_X, test_y), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, -n_out_variables:]
        inv_y2 = inv_y[:,-n_out_variables]

        rms_errors = []
        for i in range(n_out_variables):
            e = sqrt(mean_squared_error(inv_y[:, i], inv_pred[:, i]))
            rms_errors.append(e)

        rmse = sqrt(mean_squared_error(inv_y, inv_pred))
        print('Test RMSE: %.3f' % rmse)

        result_df = pandas.DataFrame(np.stack((inv_y2, inv_pred2), axis=1), columns=["Actual", "Pred"], index=test.index.values)
        result_df.to_csv('Daily_Preds_First_Day_' + str(repo_id) + '.csv')
        result_df.plot()
        pyplot.savefig('first_day_plot_repo' + str(repo_id) + '.png')

        results_by_week = result_df.resample('W', how='sum')
        results_by_week.to_csv('Weekly_Preds_' + str(repo_id) + '.csv')
        results_by_week.plot()
        pyplot.savefig('weekly_plot' + str(repo_id) + '.png')

if __name__ == '__main__':
    f = Forecasting()
    daily_repo_stats = DatabaseService().get_daily_repo_stats_by_day(1)

    data_df = pandas.DataFrame(daily_repo_stats)
    data_df.drop('repo_id', axis=1, inplace=True)
    data_df['date'] = pandas.to_datetime(data_df['date'])
    data_df.set_index("date", inplace=True)
    input_features = ['star_count', 'commit_count', 'opened_issue_count', 'closed_issue_count']
    output_features = ['star_count']

    f.forecast_stars_of_repo(1, data_df, input_features, output_features)