from keras.models import Sequential
from keras.layers import *
import pandas
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import timedelta
import os

from github_trends.services.database_service import DatabaseService

output_dir = os.path.join(os.path.dirname(__file__), "output")


class Forecasting:
    def __init__(self):
        self.db_service = DatabaseService()
        self.scaled_errors = []
        self.errors = []
        self.errors_by_day = []
        self.aggregated_relative_errors = []
        self.aggregated_absolute_errors = []
        self.epochs = 50
        self.units = 500

    def populate_non_existing_dates(self, df, columns):
        date_index = df.index
        dates = pandas.date_range(date_index.min(), date_index.max())
        dates_df = pandas.DataFrame({'drop_' + column: 0 for column in columns}, index=dates)
        merged_df = pandas.merge(df, dates_df, left_index=True, right_index=True, how='outer')

        for (ix, row) in merged_df.iterrows():
            if ix not in df.index:
                for col in df.columns:
                    merged_df.at[ix, col] = 0

        merged_df.drop(dates_df.columns, axis=1, inplace=True)
        return merged_df

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

    def __plot_boxplot_with_mean_labels(self, df, file_name):
        df.boxplot(grid=False, fontsize=8, return_type='axes', figsize=(13, 10))
        locs, labels = pyplot.xticks()
        labels = ["Day {}\n(mean: {})".format(i + 1, np.round(x, 2)) for i, x in enumerate(df.mean())]
        pyplot.xticks(locs, labels)
        pyplot.savefig(file_name + '.png', format='png')

        pyplot.clf()

    def __plot_and_add_boundaries(self, df, lower_bound, upper_bound, file_name):
        axes = df.plot(style=['-', '-'], figsize=(13, 10))
        X_points = axes.lines[0]._x
        pyplot.setp(axes.lines, lw=1.3)
        pyplot.fill_between(X_points, lower_bound, upper_bound, color='m', alpha=0.2)
        pyplot.savefig(file_name + '.png', format='png')

        pyplot.clf()

    def calculate_and_plot_relative_errors(self, pred, actual, test_dates, output_days, repo_name):
        relative_errors = []
        for col in range(output_days):  # iterate over columns (days)
            relative_error = [np.round(abs(pred[row, col] - actual[row, col]) / (actual[row, col] + 1), 5) for row in
                              range(len(actual[:, col]))]
            relative_errors.append(relative_error)

        relative_errors_df = pandas.DataFrame(np.array(relative_errors).T,
                                              columns=['Day {}'.format(x + 1) for x in range(output_days)],
                                              index=test_dates)

        aggregated_relative_error = {"repo": repo_name}
        for day in relative_errors_df:
            aggregated_relative_error["{} Mean".format(day)] = relative_errors_df[day].mean()
            aggregated_relative_error["{} Min".format(day)] = relative_errors_df[day].min()
            aggregated_relative_error["{} Max".format(day)] = relative_errors_df[day].max()
            aggregated_relative_error["{} Median".format(day)] = relative_errors_df[day].median()
            aggregated_relative_error["{} SD".format(day)] = relative_errors_df[day].std()

        self.aggregated_relative_errors.append(aggregated_relative_error)

        relative_errors_df.to_csv('output/Boxplots/relative_error_' + str(repo_name) + '.csv')
        self.__plot_boxplot_with_mean_labels(relative_errors_df, 'output/Boxplots/relative_error_{}'.format(repo_name))

        absolute_errors = []
        for col in range(output_days):  # iterate over columns (days)
            abs_error = [np.round(abs(pred[row, col] - actual[row, col]), 5) for row in
                         range(len(actual[:, col]))]
            absolute_errors.append(abs_error)

        absolute_errors_df = pandas.DataFrame(np.array(absolute_errors).T,
                                              columns=['Day {}'.format(x + 1) for x in range(output_days)],
                                              index=test_dates)

        aggregated_absolute_error = {"repo": repo_name}
        for day in relative_errors_df:
            aggregated_absolute_error["{} Mean".format(day)] = absolute_errors_df[day].mean()
            aggregated_absolute_error["{} Min".format(day)] = absolute_errors_df[day].min()
            aggregated_absolute_error["{} Max".format(day)] = absolute_errors_df[day].max()
            aggregated_absolute_error["{} Median".format(day)] = absolute_errors_df[day].median()
            aggregated_absolute_error["{} SD".format(day)] = absolute_errors_df[day].std()

        self.aggregated_absolute_errors.append(aggregated_absolute_error)

        absolute_errors_df.to_csv('output/Boxplots/absolute_error_' + str(repo_name) + '.csv')
        self.__plot_boxplot_with_mean_labels(relative_errors_df, 'output/Boxplots/absolute_error_{}'.format(repo_name))

    def evaluate_first_day_results(self, first_day_rmse, pred, actual, test_dates, repo_name):
        inv_predfirst_low = np.array(list(map(lambda x: x - first_day_rmse, pred)))
        inv_predfirst_up = np.array(list(map(lambda x: x + first_day_rmse, pred)))

        first_day_df = pandas.DataFrame(np.stack((actual, pred), axis=1),
                                        columns=["Actual", "Pred"], index=test_dates)
        first_day_df.to_csv('output/Daily_First_Day/mse_' + str(repo_name) + '.csv')

        self.__plot_and_add_boundaries(first_day_df, inv_predfirst_low, inv_predfirst_up,
                                       'output/Daily_First_Day/mse_{}'.format(repo_name))

        first_days_cumulative_df = first_day_df.cumsum()
        first_days_cumulative_df.to_csv('output/First_Day_Cumulative/mse_' + str(repo_name) + '.csv')
        first_days_cumulative_df.plot()
        pyplot.savefig('output/First_Day_Cumulative/mse_' + str(repo_name) + '.png')
        pyplot.clf()

    def evaluate_last_day_results(self, last_day_rmse, pred, actual, test_dates, repo_name):
        inv_predlast_low = np.array(list(map(lambda x: x - last_day_rmse, pred)))
        inv_predlast_up = np.array(list(map(lambda x: x + last_day_rmse, pred)))

        last_day_df = pandas.DataFrame(np.stack((actual, pred), axis=1),
                                       columns=["Actual", "Pred"], index=test_dates)
        last_day_df.to_csv('output/Daily_Last_Day/mse_' + str(repo_name) + '.csv')
        axes = last_day_df.plot(style=['-', '-'], figsize=(13, 10))
        X_points = axes.lines[0]._x
        pyplot.setp(axes.lines, lw=1.3)
        pyplot.fill_between(X_points, inv_predlast_low, inv_predlast_up, color='m', alpha=0.2)
        pyplot.savefig('output/Daily_Last_Day/mse_' + str(repo_name) + '.png')

    def evaluate_projection_sum_results(self, proj_sum_rmse, pred, actual, test_dates, repo_name):
        proj_sum_pred_low = np.array(list(map(lambda x: x - proj_sum_rmse, pred)))
        proj_sum_pred_up = np.array(list(map(lambda x: x + proj_sum_rmse, pred)))

        proj_sum_df = pandas.DataFrame(np.stack((actual, pred), axis=1),
                                       columns=["Actual", "Pred"], index=test_dates)

        proj_sum_df.to_csv(('output/Weekly_Projection_Sum/mse_' + str(repo_name) + '.csv'))
        axes = proj_sum_df.plot(style=['-', '-'], figsize=(13, 10))
        X_points = axes.lines[0]._x
        pyplot.setp(axes.lines, lw=1.3)
        pyplot.fill_between(X_points, proj_sum_pred_low, proj_sum_pred_up, color='m', alpha=0.2)
        pyplot.savefig('output/Weekly_Projection_Sum/mse_' + str(repo_name) + '.png')

        pyplot.clf()

    def forecast_stars_of_repo(self, repo_name, data_df, input_features, output_features, input_days=60, output_days=7, **kwargs):
        scaler = MinMaxScaler(feature_range=(0, 1))

        n_input_steps = input_days
        n_in_variables = len(input_features)
        n_out_variables = output_days * len(output_features)
        n_obs = n_input_steps * n_in_variables

        sequence_data = self.arrange_input_sequence(data_df, n_input_steps, n_out_variables,
                                                    input_features, output_features)

        values = sequence_data.values
        values = scaler.fit_transform(values)

        train_test_ratio = 2/3
        if 'train_test_ratio' in kwargs:
            train_test_ratio = kwargs['train_test_ratio'] if kwargs['train_test_ratio'] < 1 else train_test_ratio

        training_weeks = np.math.floor(sequence_data.shape[0] * train_test_ratio)
        headers = sequence_data.columns
        indexes = sequence_data.index.values

        train = pandas.DataFrame(values[:training_weeks, :], columns=headers, index=indexes[:training_weeks])
        test = pandas.DataFrame(values[training_weeks:, :], columns=headers, index=indexes[training_weeks:])

        train_data = pandas.DataFrame(train.values[:, :n_obs], columns=train.columns.values[:n_obs],
                                      index=train.index.values)
        train_y = pandas.DataFrame(train.values[:, n_obs:], columns=train.columns.values[n_obs:],
                                   index=train.index.values)
        test_data = pandas.DataFrame(test.values[:, :n_obs], columns=test.columns.values[:n_obs],
                                     index=test.index.values)
        test_y = pandas.DataFrame(test.values[:, n_obs:], columns=test.columns.values[n_obs:], index=test.index.values)

        # reshape to 3-D
        train_X = train_data.values.reshape((train_data.values.shape[0], n_input_steps, n_in_variables))
        test_X = test_data.values.reshape((test_data.values.shape[0], n_input_steps, n_in_variables))

        model = Sequential()
        lstm = LSTM(self.units, input_shape=(train_X.shape[1], train_X.shape[2]), activation="sigmoid")
        # rnn = SimpleRNN(500, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, activation="sigmoid")
        # model.add(rnn)
        model.add(lstm)
        model.add(Dense(n_out_variables))
        model.compile(loss='mse', optimizer='adam')

        history = model.fit(train_X, train_y, epochs=self.epochs, batch_size=32, verbose=1, validation_split=0.10,
                            shuffle=False)
        pred = model.predict(test_X, verbose=1)

        test_X = test_X.reshape((test_X.shape[0], n_obs))
        inv_pred = np.concatenate((test_X, pred), axis=1)
        inv_pred = scaler.inverse_transform(inv_pred)
        inv_pred = inv_pred[:, -n_out_variables:]
        inv_predfirst = inv_pred[:, -n_out_variables]
        inv_predlast = inv_pred[:, -1]

        inv_y = np.concatenate((test_X, test_y), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, -n_out_variables:]
        inv_yfirst = inv_y[:, -n_out_variables]
        inv_ylast = inv_y[:, -1]

        rmse = sqrt(mean_squared_error(test_y, pred))
        print('Test RMSE Scaled: %.3f' % rmse)
        self.scaled_errors.append({"repo": repo_name, "error": rmse})

        rmse = sqrt(mean_squared_error(inv_y, inv_pred))
        print('Test RMSE: %.3f' % rmse)
        self.errors.append({"repo": repo_name, "error": rmse})

        rms_errors_by_day = {"repo": repo_name}
        for col in range(inv_y.shape[1]):
            rms_error = sqrt(mean_squared_error(inv_pred[:, col], inv_y[:, col]))
            rms_errors_by_day['Day {}'.format(col + 1)] = rms_error
        self.errors_by_day.append(rms_errors_by_day)

        first_day_rmse = rms_errors_by_day['Day 1']
        last_day_rmse = rms_errors_by_day['Day {}'.format(output_days)]
        proj_sum_rmse = sum({k: v for k, v in rms_errors_by_day.items() if k.startswith('Day')}.values())

        self.calculate_and_plot_relative_errors(inv_pred, inv_y, test.index.values, output_days, repo_name)

        self.evaluate_first_day_results(first_day_rmse, inv_predfirst, inv_yfirst, test.index.values, repo_name)

        lastday_ix = test.index.values + np.timedelta64(n_out_variables - 1, 'D')
        self.evaluate_last_day_results(last_day_rmse, inv_predlast, inv_ylast, lastday_ix, repo_name)

        proj_sum_pred = np.sum(inv_pred, axis=1)
        proj_sum_y = np.sum(inv_y, axis=1)
        self.evaluate_projection_sum_results(proj_sum_rmse, proj_sum_pred, proj_sum_y, test.index.values, repo_name)


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Daily_First_Day"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Daily_Last_Day"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Weekly_Projection_Sum"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Boxplots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "RMSErrors"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "First_Day_Cumulative"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Last_Day_Cumulative"), exist_ok=True)

    f = Forecasting()
    db_service = DatabaseService()
    for i in range(1, 28):
        repo_name = db_service.get_repo_full_name_by_id(i)
        repo_name = repo_name.replace("/", "_")
        daily_repo_stats = DatabaseService().get_daily_repo_stats_by_day(i)

        data_df = pandas.DataFrame(daily_repo_stats)
        data_df.drop('repo_id', axis=1, inplace=True)
        data_df['date'] = pandas.to_datetime(data_df['date'])
        data_df.set_index("date", inplace=True)

        input_features = data_df.columns
        output_features = ['star_count']

        data_df = f.populate_non_existing_dates(data_df, columns=(input_features.union(output_features)))

        f.forecast_stars_of_repo(repo_name, data_df, input_features, output_features, input_days=60, output_days=7,
                                 train_test_ratio=(2/3))

        pandas.DataFrame(f.errors).to_csv("output/Errors.csv")
        pandas.DataFrame(f.scaled_errors).to_csv("output/Scaled_Errors.csv")
        pandas.DataFrame(f.errors_by_day).to_csv("output/Errors_By_Day.csv")
        pandas.DataFrame(f.aggregated_relative_errors).to_csv("output/Aggregated_Relative_Errors.csv")
        pandas.DataFrame(f.aggregated_absolute_errors).to_csv("output/Aggregated_Absolute_Errors.csv")
