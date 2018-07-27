from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from math import sqrt
from datetime import timedelta
import os
from collections import OrderedDict

from github_trends.services.database_service import DatabaseService

output_dir = os.path.join(os.path.dirname(__file__), "output")


class Forecasting:
    def __init__(self):
        self.db_service = DatabaseService()
        self.scaled_errors = []
        self.errors = []
        self.test_errors_by_day = []
        self.train_errors_by_day = []
        self.aggregated_relative_errors = []
        self.aggregated_absolute_errors = []
        self.aggregated_percentage_errors = []
        self.epochs = 500
        self.units = 500

    def populate_non_existing_dates(self, df, columns):
        date_index = df.index
        dates = pd.date_range(date_index.min(), date_index.max())
        dates_df = pd.DataFrame({'drop_' + column: 0 for column in columns}, index=dates)
        merged_df = pd.merge(df, dates_df, left_index=True, right_index=True, how='outer')

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
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        agg.dropna(inplace=True)

        return agg

    def __plot_boxplot_with_mean_labels(self, df, file_name):
        df.boxplot(grid=False, fontsize=8, return_type='axes', figsize=(13, 10))
        locs, labels = pyplot.xticks()
        days = [1, 7, 14, 30]
        labels = ["Day {}\n(mean: {})".format(x[0], np.round(x[1], 2)) for x in zip(days, df.mean())]
        pyplot.xticks(locs, labels)
        pyplot.savefig(file_name + '.png', format='png')

        pyplot.clf()

    def __plot_and_add_boundaries(self, df, lower_bound, upper_bound, file_name):
        axes = df.plot(style=['-', '-'], figsize=(13, 10))
        X_points = axes.lines[0]._x
        pyplot.setp(axes.lines, lw=1.3)
        # pyplot.fill_between(X_points, lower_bound, upper_bound, color='m', alpha=0.2)
        pyplot.savefig(file_name + '.png', format='png')

        pyplot.clf()

    def calculate_and_plot_relative_errors(self, pred, actual, test_dates, output_days, repo_name, suffix=""):
        filename = "{}{}".format(repo_name, "_" + suffix)
        relative_errors = np.round(np.abs(pred - actual) / (actual + 1), 3)

        relative_errors_df = pd.DataFrame(relative_errors,
                                          columns=['Day {}'.format(x + 1) for x in range(output_days)],
                                          index=test_dates)

        aggregated_relative_error = OrderedDict({"repo": repo_name})
        for day in relative_errors_df:
            aggregated_relative_error["{} Mean".format(day)] = relative_errors_df[day].mean()
            aggregated_relative_error["{} Min".format(day)] = relative_errors_df[day].min()
            aggregated_relative_error["{} Max".format(day)] = relative_errors_df[day].max()
            aggregated_relative_error["{} Median".format(day)] = relative_errors_df[day].median()
            aggregated_relative_error["{} SD".format(day)] = relative_errors_df[day].std()

        self.aggregated_relative_errors.append(aggregated_relative_error)

        relative_errors_df.to_csv('output/Boxplots/relative_error_' + filename + '.csv')
        self.__plot_boxplot_with_mean_labels(relative_errors_df[['Day 1', 'Day 7', 'Day 14', 'Day 30']],
                                             'output/Boxplots/relative_error_{}'.format(filename))

        absolute_errors = np.round(np.abs(pred - actual), 3)

        absolute_errors_df = pd.DataFrame(absolute_errors,
                                          columns=['Day {}'.format(x + 1) for x in range(output_days)],
                                          index=test_dates)

        aggregated_absolute_error = OrderedDict({"repo": repo_name})
        for day in absolute_errors_df:
            aggregated_absolute_error["{} Mean".format(day)] = absolute_errors_df[day].mean()
            aggregated_absolute_error["{} Min".format(day)] = absolute_errors_df[day].min()
            aggregated_absolute_error["{} Max".format(day)] = absolute_errors_df[day].max()
            aggregated_absolute_error["{} Median".format(day)] = absolute_errors_df[day].median()
            aggregated_absolute_error["{} SD".format(day)] = absolute_errors_df[day].std()

        self.aggregated_absolute_errors.append(aggregated_absolute_error)

        absolute_errors_df.to_csv('output/Boxplots/absolute_error_' + filename + '.csv')
        self.__plot_boxplot_with_mean_labels(absolute_errors_df[['Day 1', 'Day 7', 'Day 14', 'Day 30']],
                                             'output/Boxplots/absolute_error_{}'.format(filename))

        percentage_errors = np.round(100 * np.abs(pred - actual) / (actual + 1), 3)

        percentage_errors_df = pd.DataFrame(percentage_errors,
                                            columns=['Day {}'.format(x + 1) for x in range(output_days)],
                                            index=test_dates)

        aggregated_percentage_error = OrderedDict({"repo": repo_name})
        for day in percentage_errors_df:
            aggregated_percentage_error["{} Mean".format(day)] = percentage_errors_df[day].mean()
            aggregated_percentage_error["{} Min".format(day)] = percentage_errors_df[day].min()
            aggregated_percentage_error["{} Max".format(day)] = percentage_errors_df[day].max()
            aggregated_percentage_error["{} Median".format(day)] = percentage_errors_df[day].median()
            aggregated_percentage_error["{} SD".format(day)] = percentage_errors_df[day].std()

        self.aggregated_percentage_errors.append(aggregated_percentage_error)

        percentage_errors_df.to_csv('output/Boxplots/percentage_error_' + filename + '.csv')
        self.__plot_boxplot_with_mean_labels(percentage_errors_df[['Day 1', 'Day 7', 'Day 14', 'Day 30']],
                                             'output/Boxplots/percentage_error_{}'.format(filename))

    def evaluate_first_day_results(self, first_day_mae, pred, actual, test_dates, repo_name, suffix=""):
        filename = "{}{}".format(repo_name, "_" + suffix)

        low_bound = np.array(list(map(lambda x: x - first_day_mae, pred)))
        up_bound = np.array(list(map(lambda x: x + first_day_mae, pred)))

        first_day_df = pd.DataFrame(np.stack((actual, pred), axis=1),
                                    columns=["Actual", "Pred"], index=test_dates)
        first_day_df.to_csv('output/Daily_First_Day/mae_' + filename + '.csv', sep=";")

        self.__plot_and_add_boundaries(first_day_df, low_bound, up_bound,
                                       'output/Daily_First_Day/mae_{}'.format(filename))

        first_days_cumulative_df = first_day_df.cumsum()

        first_days_cumulative_df.to_csv('output/First_Day_Cumulative/mae_' + filename + '.csv', sep=";")
        first_days_cumulative_df.plot()
        pyplot.savefig('output/First_Day_Cumulative/mae_' + filename + '.png')
        pyplot.clf()

    def evaluate_last_day_results(self, last_day_mae, pred, actual, test_dates, repo_name, suffix=""):
        filename = "{}{}".format(repo_name, "_" + suffix)

        low_bound = np.array(list(map(lambda x: x - last_day_mae, pred)))
        up_bound = np.array(list(map(lambda x: x + last_day_mae, pred)))

        last_day_df = pd.DataFrame(np.stack((actual, pred), axis=1),
                                   columns=["Actual", "Pred"], index=test_dates)
        last_day_df.to_csv('output/Daily_Last_Day/mae_' + filename + '.csv', sep=";")
        axes = last_day_df.plot(style=['-', '-'], figsize=(13, 10))
        # X_points = axes.lines[0]._x
        pyplot.setp(axes.lines, lw=1.3)
        # pyplot.fill_between(X_points, low_bound, up_bound, color='m', alpha=0.2)
        pyplot.savefig('output/Daily_Last_Day/mae_' + filename + '.png')

        last_days_cumulative_df = last_day_df.cumsum()
        last_days_cumulative_df.to_csv('output/Last_Day_Cumulative/mae_' + filename + '.csv', sep=";")
        last_days_cumulative_df.plot()
        pyplot.savefig('output/Last_Day_Cumulative/mae_' + filename + '.png')
        pyplot.clf()

    def evaluate_first_week_results(self, pred, actual, test_dates, repo_name, suffix=""):
        filename = "{}{}".format(repo_name, "_" + suffix)

        first_week_df = pd.DataFrame(np.stack((actual, pred), axis=1),
                                     columns=["Actual", "Pred"], index=test_dates)
        first_week_df.to_csv('output/Daily_First_Week/mae_' + filename + '.csv', sep=";")
        axes = first_week_df.plot(style=['-', '-'], figsize=(13, 10))
        # X_points = axes.lines[0]._x
        pyplot.setp(axes.lines, lw=1.3)
        # pyplot.fill_between(X_points, inv_predlast_low, inv_predlast_up, color='m', alpha=0.2)
        pyplot.savefig('output/Daily_First_Week/mae_' + filename + '.png')

        last_days_cumulative_df = first_week_df.cumsum()
        last_days_cumulative_df.to_csv('output/First_Week_Cumulative/mae_' + filename + '.csv', sep=";")
        last_days_cumulative_df.plot()
        pyplot.savefig('output/First_Week_Cumulative/mae_' + filename + '.png')
        pyplot.clf()

    def evaluate_second_week_results(self, pred, actual, test_dates, repo_name, suffix=""):
        filename = "{}{}".format(repo_name, "_" + suffix)

        second_week_df = pd.DataFrame(np.stack((actual, pred), axis=1),
                                      columns=["Actual", "Pred"], index=test_dates)
        second_week_df.to_csv('output/Daily_Second_Week/mae_' + filename + '.csv', sep=";")
        axes = second_week_df.plot(style=['-', '-'], figsize=(13, 10))
        # X_points = axes.lines[0]._x
        pyplot.setp(axes.lines, lw=1.3)
        # pyplot.fill_between(X_points, inv_predlast_low, inv_predlast_up, color='m', alpha=0.2)
        pyplot.savefig('output/Daily_Second_Week/mae_' + filename + '.png')

        last_days_cumulative_df = second_week_df.cumsum()
        last_days_cumulative_df.to_csv('output/Second_Week_Cumulative/mae_' + filename + '.csv', sep=";")
        last_days_cumulative_df.plot()
        pyplot.savefig('output/Second_Week_Cumulative/mae_' + filename + '.png')
        pyplot.clf()

    def evaluate_projection_sum_results(self, proj_sum_mae, pred, actual, test_dates, repo_name, suffix=""):
        filename = "{}{}".format(repo_name, "_" + suffix)

        low_bound = np.array(list(map(lambda x: x - proj_sum_mae, pred)))
        up_bound = np.array(list(map(lambda x: x + proj_sum_mae, pred)))

        proj_sum_df = pd.DataFrame(np.stack((actual, pred), axis=1),
                                   columns=["Actual", "Pred"], index=test_dates)

        proj_sum_df.to_csv('output/Weekly_Projection_Sum/mae_' + filename + '.csv', sep=";")
        axes = proj_sum_df.plot(style=['-', '-'], figsize=(13, 10))
        X_points = axes.lines[0]._x
        pyplot.setp(axes.lines, lw=1.3)
        # pyplot.fill_between(X_points, low_bound, up_bound, color='m', alpha=0.2)
        pyplot.savefig('output/Weekly_Projection_Sum/mae_' + filename + '.png')

        pyplot.clf()

    def __get_inverse_scaled(self, scaler, data, pred, n_out_variables):
        inv = np.concatenate((data, pred), axis=1)
        inv = scaler.inverse_transform(inv)
        return inv[:, -n_out_variables:]

    def __decompose_predictions_to_days(self, pred_matrix, columns):
        values_to_return = []
        for col in columns:
            values_to_return.append(pred_matrix[:, col])

        return tuple(values_to_return)

    def __calculate_daily_errors(self, repo_name, actual, pred):
        mae_by_day = OrderedDict({"repo": repo_name})
        for col in range(actual.shape[1]):
            test_mae = mean_absolute_error(pred[:, col], actual[:, col])
            mae_by_day['Day {}'.format(col + 1)] = np.round(test_mae, 3)

        return mae_by_day

    def __plot_complete_results(self, train_df, test_df, filename, indices_to_plot):
        full_actual_df = pd.concat([train_df.filter(like='actual_'),
                                    test_df.filter(like='actual_')], axis=0)
        pred_cols = train_df.filter(like='pred_').columns
        train_pred_df = train_df[pred_cols]
        test_pred_df = test_df[pred_cols]

        complete_pred_df = pd.concat([train_pred_df.add_prefix("train_"), test_pred_df.add_prefix("test_")], axis=0)

        full_df = pd.concat([full_actual_df, complete_pred_df], axis=1)
        full_df.to_csv("output/Predictions/{}.csv".format(filename))

        for i in indices_to_plot:
            cols = ['actual_' + str(i),
                    'train_pred_' + str(i),
                    'test_pred_' + str(i)]

            full_df[cols].plot(figsize=(20, 16))
            pyplot.savefig("output/Predictions/{}_Day_{}.png".format(filename, i))


    def forecast_stars_of_repo(self, repo_name, data_df, input_features, output_features, input_days=60, output_days=7,
                               **kwargs):
        scaler = MinMaxScaler(feature_range=(0, 1))

        n_input_steps = input_days
        n_in_variables = len(input_features)
        n_out_variables = output_days * len(output_features)
        n_obs = n_input_steps * n_in_variables

        sequence_data = self.arrange_input_sequence(data_df, n_input_steps, n_out_variables,
                                                    input_features, output_features)

        values = sequence_data.values
        values = scaler.fit_transform(values)

        train_test_ratio = 2 / 3
        if 'train_test_ratio' in kwargs:
            train_test_ratio = kwargs['train_test_ratio'] if kwargs['train_test_ratio'] < 1 else train_test_ratio

        training_weeks = np.math.floor(sequence_data.shape[0] * train_test_ratio)
        headers = sequence_data.columns
        indexes = sequence_data.index.values

        train = pd.DataFrame(values[:training_weeks, :], columns=headers, index=indexes[:training_weeks])
        test = pd.DataFrame(values[training_weeks:, :], columns=headers, index=indexes[training_weeks:])

        train_data = pd.DataFrame(train.values[:, :n_obs], columns=train.columns.values[:n_obs],
                                  index=train.index.values)
        train_y = pd.DataFrame(train.values[:, n_obs:], columns=train.columns.values[n_obs:],
                               index=train.index.values)
        test_data = pd.DataFrame(test.values[:, :n_obs], columns=test.columns.values[:n_obs],
                                 index=test.index.values)
        test_y = pd.DataFrame(test.values[:, n_obs:], columns=test.columns.values[n_obs:], index=test.index.values)

        # reshape to 3-D
        train_X = train_data.values.reshape((train_data.values.shape[0], n_input_steps, n_in_variables))
        test_X = test_data.values.reshape((test_data.values.shape[0], n_input_steps, n_in_variables))

        early_stopping_cb = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

        model = Sequential()
        # rnn = SimpleRNN(500, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, activation="sigmoid")
        # model.add(rnn)
        model.add(Dense(n_in_variables * n_input_steps, input_shape=(train_X.shape[1], train_X.shape[2])))
        lstm = CuDNNLSTM(self.units)  # if CUDA, use CuDNNLSTM
        model.add(CuDNNLSTM(self.units, return_sequences=True))  # if CUDA, us CuDNNLSTM
        model.add(lstm)
        model.add(Dropout(0.5))
        model.add(Dense(n_out_variables))
        model.summary()
        model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])

        history = model.fit(train_X, train_y, epochs=self.epochs, batch_size=32, verbose=1, validation_split=0.10,
                            shuffle=False, callbacks=[early_stopping_cb])

        test_pred = model.predict(test_X, verbose=1)
        train_pred = model.predict(train_X, verbose=1)

        test_X = test_X.reshape((test_X.shape[0], n_obs))
        train_X = train_X.reshape((train_X.shape[0], n_obs))

        # invert scaling back to originals
        inv_pred_test = self.__get_inverse_scaled(scaler, test_X, test_pred, n_out_variables)
        inv_test_y = self.__get_inverse_scaled(scaler, test_X, test_y, n_out_variables)

        inv_pred_train = self.__get_inverse_scaled(scaler, train_X, train_pred, n_out_variables)
        inv_train_y = self.__get_inverse_scaled(scaler, train_X, train_y, n_out_variables)

        # Export Predictions
        columns = range(1, inv_train_y.shape[1]+1)
        train_predictions = pd.concat([pd.DataFrame(inv_train_y, columns=columns).add_prefix("actual_"),
                                       pd.DataFrame(inv_pred_train, columns=columns).add_prefix("pred_")],
                                      axis=1)
        train_predictions.index = train.index
        train_predictions.to_csv('output/Predictions/{}_train_predictions.csv'.format(repo_name))

        test_predictions = pd.concat([pd.DataFrame(inv_test_y, columns=columns).add_prefix("actual_"),
                                      pd.DataFrame(inv_pred_test, columns=columns).add_prefix("pred_")], axis=1)
        test_predictions.index = test.index
        test_predictions.to_csv('output/Predictions/{}_test_predictions.csv'.format(repo_name))

        test_mae = mean_absolute_error(test_y, test_pred)
        test_mape = np.average(np.abs((test_y - test_pred) / test_y)) * 100
        train_mae = mean_absolute_error(train_y, train_pred)
        train_mape = np.average(np.abs((train_y - train_pred) / train_y)) * 100
        print('Test MAE Scaled: %.3f' % test_mae)
        print('Train MAE Scaled: %.3f' % train_mae)

        self.scaled_errors.append({"repo": repo_name,
                                   "test_mae": np.round(test_mae, 3), "test_mape": np.round(test_mape, 3),
                                   "train_mae": np.round(train_mae, 3), "train_mape": np.round(train_mape, 3)})

        test_mae = mean_absolute_error(inv_test_y, inv_pred_test)
        test_mape = np.average(np.abs((inv_test_y - inv_pred_test) / inv_test_y)) * 100
        train_mae = mean_absolute_error(inv_train_y, inv_pred_train)
        train_mape = np.average(np.abs((inv_train_y - inv_pred_train) / inv_train_y)) * 100

        print('Test MAE: %.3f' % test_mae)
        print('Train MAE: %.3f' % train_mae)

        self.errors.append({"repo": repo_name,
                            "test_mae": np.round(test_mae, 3), "test_mape": np.round(test_mape, 3),
                            "train_mae": np.round(train_mae, 3), "train_mape": np.round(train_mape, 3)})

        test_mae_by_day = self.__calculate_daily_errors(repo_name, inv_test_y, inv_pred_test)
        train_mae_by_day = self.__calculate_daily_errors(repo_name, inv_train_y, inv_pred_train)

        self.test_errors_by_day.append(test_mae_by_day)
        self.train_errors_by_day.append(train_mae_by_day)

        evaluation_criterias = [
            {
                "type": "test",
                "error_values": test_mae_by_day,
                "index": test.index.values,
                "data": {
                    "pred": inv_pred_test,
                    "actual": inv_test_y
                }
            },
            {
                "type": "train",
                "error_values": train_mae_by_day,
                "index": train.index.values,
                "data": {
                    "pred": inv_pred_train,
                    "actual": inv_train_y
                }
            }
        ]

        for criterion in evaluation_criterias:
            pred = criterion["data"]["pred"]
            actual = criterion["data"]["actual"]

            first_day_mae = criterion["error_values"]['Day 1']
            last_day_mae = criterion["error_values"]['Day {}'.format(output_days)]
            proj_sum_mae = sum({k: v
                                for k, v in criterion["error_values"].items()
                                if k.startswith('Day')}.values())

            pred_first, pred_firstweek, pred_secondweek, pred_last = \
                self.__decompose_predictions_to_days(pred, [0, 6, 13, -1])

            actual_first, actual_firstweek, actual_secondweek, actual_last = \
                self.__decompose_predictions_to_days(actual, [0, 6, 13, -1])

            self.calculate_and_plot_relative_errors(pred, actual, criterion["index"], output_days,
                                                    repo_name, criterion["type"])

            self.evaluate_first_day_results(first_day_mae, pred_first, actual_first, criterion["index"],
                                            repo_name, criterion["type"])

            lastday_ix = criterion["index"] + np.timedelta64(n_out_variables - 1, 'D')
            self.evaluate_last_day_results(last_day_mae, pred_last, actual_last, lastday_ix, repo_name, criterion["type"])

            first_week_ix = criterion["index"] + np.timedelta64(6, 'D')
            self.evaluate_first_week_results(pred_firstweek, actual_firstweek, first_week_ix,
                                             repo_name, criterion["type"])

            second_week_ix = criterion["index"] + np.timedelta64(13, 'D')
            self.evaluate_second_week_results(pred_secondweek, actual_secondweek, second_week_ix,
                                              repo_name, criterion["type"])

            proj_sum_pred = np.sum(pred, axis=1)
            proj_sum_y = np.sum(actual, axis=1)
            self.evaluate_projection_sum_results(proj_sum_mae, proj_sum_pred, proj_sum_y, criterion["index"], repo_name, criterion["type"])

        ''' Plot train & test values together '''
        self.__plot_complete_results(train_predictions, test_predictions, repo_name, [1, 7, 14, 30])


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Predictions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Daily_First_Day"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Daily_Last_Day"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Daily_First_Week"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Daily_Second_Week"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Weekly_Projection_Sum"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Boxplots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "RMSErrors"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "First_Day_Cumulative"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Last_Day_Cumulative"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "First_Week_Cumulative"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Second_Week_Cumulative"), exist_ok=True)

    f = Forecasting()
    db_service = DatabaseService()

    '''
    ['d3/d3', 'matplotlib/matplotlib', 'ecomfe/echarts', 'mozilla/metrics-graphics',
                     'jacomyal/sigma.js', 'chartjs/Chart.js', 'gionkunz/chartist-js', 'Leaflet/Leaflet']
    '''
    allowed_repos = ['d3/d3', 'ecomfe/echarts', 'chartjs/Chart.js', 'gionkunz/chartist-js', 'Leaflet/Leaflet']
    for i in range(1, 28):
        repo_name = db_service.get_repo_full_name_by_id(i)
        if repo_name not in allowed_repos:
            continue
        repo_name = repo_name.replace("/", "_")
        daily_repo_stats = DatabaseService().get_daily_repo_stats_by_day(i)

        data_df = pd.DataFrame(daily_repo_stats)
        data_df.drop('repo_id', axis=1, inplace=True)
        data_df['date'] = pd.to_datetime(data_df['date'])
        data_df.set_index("date", inplace=True)
        input_features = data_df.columns
        input_features = pd.Index(['weighted_star_count', 'weighted_commit_count', 'weighted_opened_count',
                                   'weighted_closed_issue_count', 'weighted_fork_count', 'weighted_release_count',
                                   'star_count'])
        output_features = ['star_count']

        data_df = f.populate_non_existing_dates(data_df, columns=(input_features.union(output_features)))

        print(repo_name)
        f.forecast_stars_of_repo(repo_name, data_df, input_features, output_features, input_days=180, output_days=30,
                                 train_test_ratio=0.70)

        pd.DataFrame(f.errors).to_csv("output/Errors.csv", sep=";")
        pd.DataFrame(f.scaled_errors).to_csv("output/Scaled_Errors.csv", sep=";")
        pd.DataFrame(f.test_errors_by_day).to_csv("output/Test_Errors_By_Day.csv", sep=";")
        pd.DataFrame(f.aggregated_relative_errors).to_csv("output/Aggregated_Relative_Errors.csv", sep=";")
        pd.DataFrame(f.aggregated_absolute_errors).to_csv("output/Aggregated_Absolute_Errors.csv", sep=";")
        pd.DataFrame(f.train_errors_by_day).to_csv("output/Train_Errors_By_Day.csv", sep=";")
