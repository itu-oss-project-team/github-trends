import os
from datetime import datetime

import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import pymysql
from isoweek import Week

from github_trends import secret_config
from github_trends.services.database_service import DatabaseService

date_range_start = datetime(2014, 1, 1)
date_range_end = datetime(2016, 12, 31)
output_dir = os.path.join(os.path.dirname(__file__), "output")


def get_shape_for_release_date(release_date):
    return {
        'type': 'line',
        'x0': release_date,
        'x1': release_date,
        'line': {
            'color': 'rgb(220,220,220)',
            'width': 1,
            'dash': 'dot',
        },
    }


def timeline_plot(figure_data, repo_name, group_name, release_dates):
    shapes = [get_shape_for_release_date(release_date) for release_date in release_dates]
    layout = dict(
        title=repo_name,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='6m',
                         step='month',
                         stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(),
            type='date'
        ),

        yaxis=dict(
            # type='log',
            autorange=True
        ),

        shapes=shapes
    )

    fig = dict(data=figure_data, layout=layout)
    py.offline.plot(fig, filename='output/' + group_name + '/' + repo_name.replace('/', '-') + '.html', auto_open=False)


def first_date_of_week_of_date(_date):
    return Week.withdate(_date).monday()


def first_date_of_month_of_date(_date):
    return _date.replace(day=1)


def group_by_and_draw_graph(daily_commits, daily_issues, daily_stars, daily_forks, repo_name, group_function,
                            group_name, release_dates, cumulative = False):
    grouped_commits = pd.Series(index=daily_commits.index.map(group_function),
                                data=daily_commits['commit_count'].values).groupby(level=0).sum()
    if cumulative:
        grouped_commits = grouped_commits.sort_index().cumsum()

    grouped_issue_openings = pd.Series(index=daily_issues.index.map(group_function),
                                       data=daily_issues['opened_count'].values).groupby(level=0).sum()
    if cumulative:
        grouped_issue_openings = grouped_issue_openings.sort_index().cumsum()

    grouped_issue_closings = pd.Series(index=daily_issues.index.map(group_function),
                                       data=daily_issues['closed_count'].values).groupby(level=0).sum()
    if cumulative:
        grouped_issue_closings = grouped_issue_closings.sort_index().cumsum()

    grouped_resolution_durations = pd.Series(index=daily_issues.index.map(group_function),
                                             data=daily_issues['total_resolution_hours'].values).groupby(level=0).sum()

    grouped_stars = pd.Series(index=daily_stars.index.map(group_function),
                              data=daily_stars['star_count'].values).groupby(level=0).sum()
    if cumulative:
        grouped_stars = grouped_stars.sort_index().cumsum()

    grouped_forks = pd.Series(index=daily_forks.index.map(group_function),
                              data=daily_forks['fork_count'].values).groupby(level=0).sum()
    if cumulative:
        grouped_forks = grouped_forks.sort_index().cumsum()

    trace_commits = go.Scatter(
        x=grouped_commits.index,
        y=grouped_commits.values,
        name="{} Commits".format(group_name))

    trace_issue_openings = go.Scatter(
        x=grouped_issue_openings.index,
        y=grouped_issue_openings.values,
        name="{} Issue Openings".format(group_name))

    trace_issue_closings = go.Scatter(
        x=grouped_issue_closings.index,
        y=grouped_issue_closings.values,
        name="{} Issue Closings".format(group_name))

    trace_stars = go.Scatter(
        x=grouped_stars.index,
        y=grouped_stars.values,
        name="{} Stars".format(group_name))

    trace_forks = go.Scatter(
        x=grouped_forks.index,
        y=grouped_forks.values,
        name="{} Forks".format(group_name))

    grouped_data = [trace_commits, trace_issue_openings, trace_issue_closings, trace_stars, trace_forks]
    timeline_plot(grouped_data, repo_name, group_name, release_dates)
    '''
    grouped_stats = pd.concat(
        [grouped_commits, grouped_issue_openings, grouped_issue_closings, grouped_resolution_durations],
        axis=1)
    
    py.tools.set_credentials_file(username='karpatkubilay', api_key='FTHyTbxfuyZHixa9gopR')
    grouped_stats.iplot(kind='scatter', filename=repo['owner'] + '-' + repo['name'] + '.html')
    '''


def main():
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Daily"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Weekly"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Monthly"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Daily-Cumulative"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Weekly-Cumulative"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Monthly-Cumulative"), exist_ok=True)

    mysql_config = secret_config['mysql']
    mysql_config = mysql_config
    conn = pymysql.connect(host=mysql_config['host'], port=mysql_config['port'], db=mysql_config['db'],
                           user=mysql_config['user'], passwd=mysql_config['passwd'], charset='utf8mb4',
                           use_unicode=True)

    db_service = DatabaseService()

    repos = db_service.execute_select_query("SELECT * FROM repos", None)

    for repo in repos:
        daily_commits_query = '''
            SELECT date, commit_count 
            FROM daily_repo_commits 
            WHERE repo_id = %s AND DATE BETWEEN %s AND %s
        '''

        # Read daily commit counts of repo from DB
        daily_commits = pd.read_sql(sql=daily_commits_query, con=conn,
                                    params=[repo['id'], date_range_start, date_range_end], index_col="date")

        daily_repo_issues_query = '''
            SELECT date, opened_count, closed_count, avg_resolution_sec 
            FROM daily_repo_issues
            WHERE repo_id = %s AND DATE BETWEEN %s AND %s
        '''

        daily_issues = pd.read_sql(sql=daily_repo_issues_query, con=conn,
                                   params=[repo['id'], date_range_start, date_range_end], index_col="date")
        # We can't calculate monthly avg from daily avg we need daily sums (daily_avg * daily_closed = daily_sum)
        daily_issues['total_resolution_hours'] = daily_issues['avg_resolution_sec'] * daily_issues[
            'closed_count'] / 60  # Convert sec to min

        daily_stars_query = '''
            SELECT date, star_count 
            FROM daily_repo_stars
            WHERE repo_id = %s AND DATE BETWEEN %s AND %s
        '''

        daily_stars = pd.read_sql(sql=daily_stars_query, con=conn,
                                  params=[repo['id'], date_range_start, date_range_end], index_col="date")

        daily_forks_query = '''
            SELECT date, fork_count 
            FROM daily_repo_forks
            WHERE repo_id = %s AND DATE BETWEEN %s AND %s
        '''

        daily_forks = pd.read_sql(sql=daily_forks_query, con=conn,
                                  params=[repo['id'], date_range_start, date_range_end], index_col="date")

        release_dates_query = '''
            SELECT date 
            FROM daily_repo_releases 
            WHERE repo_id = %s AND DATE BETWEEN %s AND %s
            GROUP BY date 
        '''

        release_dates = db_service.execute_select_query(release_dates_query,
                                                        [repo['id'], date_range_start, date_range_end])

        release_dates = [release_date['date'] for release_date in release_dates]

        group_by_and_draw_graph(daily_commits, daily_issues, daily_stars, daily_forks, repo['full_name'], lambda x: x,
                                "Daily", release_dates)
        group_by_and_draw_graph(daily_commits, daily_issues, daily_stars, daily_forks, repo['full_name'],
                                first_date_of_month_of_date, "Monthly", release_dates)
        group_by_and_draw_graph(daily_commits, daily_issues, daily_stars, daily_forks, repo['full_name'],
                                first_date_of_week_of_date, "Weekly", release_dates)


        group_by_and_draw_graph(daily_commits, daily_issues, daily_stars, daily_forks, repo['full_name'], lambda x: x,
                                "Daily-Cumulative", release_dates, cumulative=True)
        group_by_and_draw_graph(daily_commits, daily_issues, daily_stars, daily_forks, repo['full_name'],
                                first_date_of_month_of_date, "Monthly-Cumulative", release_dates, cumulative=True)
        group_by_and_draw_graph(daily_commits, daily_issues, daily_stars, daily_forks, repo['full_name'],
                                first_date_of_week_of_date, "Weekly-Cumulative", release_dates, cumulative=True)

main()
