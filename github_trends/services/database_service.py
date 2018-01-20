from collections import OrderedDict

import pymysql

from github_trends import category_repos
from github_trends import secret_config


class DatabaseService:
    def __init__(self):
        self.__mysql_config = secret_config['mysql']

    def __get_connection(self):
        mysql_config = self.__mysql_config
        conn = pymysql.connect(host=mysql_config['host'], port=mysql_config['port'], db=mysql_config['db'],
                               user=mysql_config['user'], passwd=mysql_config['passwd'], charset='utf8mb4',
                               use_unicode=True)
        return conn

    def __executemany_insert_query(self, query, data):
        conn = self.__get_connection()
        dict_cursor = conn.cursor(pymysql.cursors.DictCursor)

        dict_cursor.executemany(query, data)
        conn.commit()
        dict_cursor.close()

    def __insert_query(self, query, data):
        conn = self.__get_connection()
        dict_cursor = conn.cursor(pymysql.cursors.DictCursor)

        dict_cursor.execute(query, data)
        conn.commit()
        dict_cursor.close()

    def execute_select_query(self, query, data):
        conn = self.__get_connection()
        dict_cursor = conn.cursor(pymysql.cursors.DictCursor)

        dict_cursor.execute(query, data)
        values = dict_cursor.fetchall()
        dict_cursor.close()
        return values

    def get_repo_id_by_full_name(self, full_name):
        query = '''SELECT id FROM repos
                   WHERE full_name = %s
                '''

        repo_id_list = self.execute_select_query(query, full_name)
        return repo_id_list[0]["id"]

    def get_repo_full_name_by_id(self, id):
        query = '''SELECT full_name FROM repos
                   WHERE id = %s
                '''

        repo_id_list = self.execute_select_query(query, id)
        return repo_id_list[0]["full_name"]

    def get_repo_id_by_name_and_owner(self, owner, name):
        query = '''SELECT id FROM repos
                   WHERE owner = %s AND name = %s
                '''

        repo_id_list = self.execute_select_query(query, (owner, name))
        return repo_id_list[0]["id"]

    def save_daily_commits_of_repo(self, owner, name, date_commit_dict):
        repo_id = self.get_repo_id_by_name_and_owner(owner, name)
        commit_data = [(repo_id, k, v) for (k, v) in date_commit_dict.items()]

        query = ''' INSERT IGNORE daily_repo_commits (repo_id, date, commit_count) 
                    VALUES (%s, %s, %s ) '''

        self.__executemany_insert_query(query, commit_data)

    def save_daily_issues_of_repo(self, owner, name, date_issue_dict, date_user_dict):
        repo_id = self.get_repo_id_by_name_and_owner(owner, name)

        issue_data = [(repo_id, date, issue["opened"], issue["closed"], issue["avg_duration"])
                      for (date, issue) in date_issue_dict.items()]
        date_issue_query = ''' INSERT IGNORE daily_repo_issues (repo_id, date, opened_count, closed_count, avg_resolution_sec) 
                    VALUES ( %s, %s, %s, %s, %s) '''
        self.__executemany_insert_query(date_issue_query, issue_data)

        user_data = [(repo_id, login, date, user["opened"], user["closed"])
                     for (date, user_dict) in date_user_dict.items()
                     for (login, user) in user_dict.items()
                     if login is not None]
        date_user_query = ''' INSERT IGNORE daily_repo_issue_activities (repo_id, login, date, open_count, closed_count) 
                    VALUES ( %s, %s, %s, %s, %s) '''
        self.__executemany_insert_query(date_user_query, user_data)

    def save_daily_stars_of_repo(self, owner, name, date_star_dict):
        repo_id = self.get_repo_id_by_name_and_owner(owner, name)
        star_data = [(repo_id, k, v) for (k, v) in date_star_dict.items()]

        query = ''' INSERT IGNORE daily_repo_stars (repo_id, date, star_count) 
                    VALUES ( %s, %s, %s) '''

        self.__executemany_insert_query(query, star_data)

    def save_daily_forks_of_repo(self, owner, name, date_fork_dict):
        repo_id = self.get_repo_id_by_name_and_owner(owner, name)
        fork_data = [(repo_id, k, v) for (k, v) in date_fork_dict.items()]

        query = ''' INSERT IGNORE daily_repo_forks (repo_id, date, fork_count) 
                    VALUES (%s, %s, %s ) '''

        self.__executemany_insert_query(query, fork_data)

    def save_daily_contributions_of_repo(self, owner, name, date_user_contribution_dict):
        repo_id = self.get_repo_id_by_name_and_owner(owner, name)
        contribution_data = [(repo_id, u, d, c) for d, v in date_user_contribution_dict.items() for u, c in v.items()]

        query = ''' INSERT IGNORE daily_repo_contributions (repo_id, login, date, commit_count) 
                    VALUES (%s, %s, %s, %s ) '''

        try:
            self.__executemany_insert_query(query, contribution_data)
        except:
            pass

    def save_daily_releases_of_repo(self, owner, name, date_release_dict):
        repo_id = self.get_repo_id_by_name_and_owner(owner, name)
        release_data = [(repo_id, k, v) for (k, v) in date_release_dict.items()]

        query = ''' INSERT IGNORE daily_repo_releases (repo_id, date, release_count)
                    VALUES (%s, %s, %s) '''

        try:
            self.__executemany_insert_query(query, release_data)
        except:
            pass

    def save_daily_so_stats_of_repo(self, full_name, stats):
        repo_id = self.get_repo_id_by_full_name(full_name)
        data = [(repo_id, stat['date'], stat['question_count'], stat['view_sum'], stat['score_sum'], stat['answer_sum'])
                for stat in stats]

        query = ''' 
            INSERT INTO daily_stackoverflow_questions (repo_id, date, question_count, view_sum, score_sum, answer_sum)
                    VALUES ( %s, %s, %s, %s, %s, %s )
        '''
        try:
            self.__executemany_insert_query(query, data)
        except:
            pass

    def save_daily_so_stats_of_repo2(self, full_name, stats):
        query = ''' 
            INSERT INTO daily_stackoverflow_questions (repo_id, date, question_count, view_sum, score_sum, answer_sum)
                    VALUES ({}, %s, %s, %s, %s, %s ) 
        '''.format(self.get_repo_id_by_full_name(full_name))
        try:
            self.__executemany_insert_query(query, stats)
        except:
            pass

    def save_categories_and_repos(self):
        categories = category_repos.keys()
        query = ''' INSERT INTO categories (name) 
                    VALUES ( %s ) '''

        self.__executemany_insert_query(query, categories)

        for category, repos in category_repos.items():
            filtered_repos = [(repo['full_name'], repo['name'], repo['owner']) for repo in repos]

            query = ''' INSERT INTO repos (full_name, name, owner) 
                        VALUES ( %s, %s, %s ) '''

            self.__executemany_insert_query(query, filtered_repos)

    def get_developers(self):
        query = ''' SELECT DISTINCT login FROM daily_repo_contributions '''

        developers = self.execute_select_query(query, None)

        return developers

    def save_commits(self, owner, name, commit_list):
        repo_id = self.get_repo_id_by_name_and_owner(owner, name)

        commit_tuple_list = list(map(lambda x: (x["date"], x["login"], repo_id), commit_list))
        query = ''' INSERT INTO commits(date, login, repo_id)
                    VALUES (%s, %s, %s)'''

        self.__executemany_insert_query(query, commit_tuple_list)

    def save_stars(self, owner, name, star_list):
        repo_id = self.get_repo_id_by_name_and_owner(owner, name)

        star_tuple_list = list(map(lambda x: (x["date"], x["login"], repo_id), star_list))
        query = ''' INSERT INTO stars(date, login, repo_id)
                    VALUES (%s, %s, %s)'''

        self.__executemany_insert_query(query, star_tuple_list)

    def save_forks(self, owner, name, fork_list):
        repo_id = self.get_repo_id_by_name_and_owner(owner, name)

        fork_tuple_list = list(map(lambda x: (x["date"], x["login"], repo_id), fork_list))
        query = ''' INSERT INTO forks(date, login, repo_id)
                    VALUES (%s, %s, %s)'''

        self.__executemany_insert_query(query, fork_tuple_list)

    def save_issues(self, owner, name, issue_list):
        repo_id = self.get_repo_id_by_name_and_owner(owner, name)

        issue_tuple_list = list(map(lambda x: (
            x["opened_date"].date() if x["opened_date"] is not None else None,
            x["closed_date"].date() if x["closed_date"] is not None else None,
            x["reporter"], x["resolver"], repo_id, x["resolution_duration"]), issue_list))

        query = ''' INSERT INTO issues(opened_date, resolved_date, reporter, resolver, repo_id, resolution_duration_sec)
                    VALUES (%s, %s, %s, %s, %s, %s) '''

        self.__executemany_insert_query(query, issue_tuple_list)

    def save_releases(self, owner, name, release_list):
        repo_id = self.get_repo_id_by_name_and_owner(owner, name)

        release_tuple_list = list(map(lambda x: (x["date"], x["login"], repo_id), release_list))
        query = ''' INSERT INTO releases(date, login, repo_id)
                    VALUES (%s, %s, %s) '''

        self.__executemany_insert_query(query, release_tuple_list)

    def get_cumulative_commits_of_a_user(self, login, date):
        query = '''
            SELECT DISTINCT 
            C.date, 
            (SELECT count(*) FROM commits innerC WHERE innerC.date <= C.date AND innerC.login = C.login) AS NumberOfCommits
            FROM `commits` C
            WHERE login = %s AND C.date <= %s
            ORDER BY C.date DESC
                '''

        commit_list = self.execute_select_query(query, (login, date))
        commit_dict = OrderedDict(map(lambda x: (x["date"], x["NumberOfCommits"]), commit_list))
        return commit_dict

    def get_cumulative_opened_issues_of_a_user(self, login, date):
        query = '''
            SELECT DISTINCT 
            I.opened_date,
            (SELECT count(*) FROM issues innerI WHERE innerI.opened_date <= I.opened_date AND innerI.reporter = I.reporter) AS NumberOfIssuesOpened                
            FROM issues I
            WHERE I.reporter = %s AND I.opened_date <= %s
            ORDER BY I.opened_date DESC
                '''

        opened_issue_list = self.execute_select_query(query, (login, date))
        opened_issue_dict = OrderedDict(map(lambda x: (x["opened_date"], x["NumberOfIssuesOpened"]), opened_issue_list))
        return opened_issue_dict

    def get_cumulative_closed_issues_of_a_user(self, login, date):
        query = '''
            SELECT DISTINCT 
            I.resolved_date,
            (SELECT count(*) FROM issues innerI WHERE innerI.resolved_date <= I.resolved_date AND innerI.resolver = I.resolver) AS NumberOfIssuesClosed
            FROM issues I
            WHERE I.resolver = %s AND I.resolved_date <= %s
            ORDER BY I.resolved_date DESC                        
              '''

        opened_issue_list = self.execute_select_query(query, (login, date))
        opened_issue_dict = OrderedDict(map(lambda x: (x["resolved_date"], x["NumberOfIssuesClosed"]), opened_issue_list))
        opened_issue_list = self.execute_select_query(query, (login, date))
        opened_issue_dict = OrderedDict(
            map(lambda x: (x["resolved_date"], x["NumberOfIssuesClosed"]), opened_issue_list))
        return opened_issue_dict

    def get_number_of_starred_repos_of_a_user(self, login, date):
        query = '''
                SELECT
                    date, 
                    (SELECT count(*) FROM stars WHERE date <= S.date AND login = S.login) AS StarCount
                FROM `stars` S
                WHERE login = %s AND date <= %s
                GROUP BY date, login
                ORDER BY date DESC
                '''

        star_list = self.execute_select_query(query, (login, date))
        star_dict = OrderedDict(map(lambda x: (x["date"], x["StarCount"]), star_list))
        return star_dict

    def get_number_of_forked_repos_of_a_user(self, login, date):
        query = '''
                SELECT 
                    date, 
                    (SELECT count(*) FROM forks WHERE date <= F.date AND login = F.login) AS ForkCount
                FROM `forks` F
                WHERE login = %s AND date <= %s
                GROUP BY date, login  
                ORDER BY date DESC
                '''

        fork_list = self.execute_select_query(query, (login, date))
        fork_dict = OrderedDict(map(lambda x: (x["date"], x["ForkCount"]), fork_list))
        return fork_dict

    def get_number_of_contributed_repos_of_a_user(self, login, date):
        query = '''
                SELECT S.date, 
                @rank:=@rank-1 AS NumberOfContributedRepos 
                FROM (SELECT 
                min(date) AS date
                FROM commits
                WHERE commits.login = %s AND date <= %s
                GROUP BY repo_id) S,
                (SELECT @rank:= (SELECT Count(*) FROM 
                (SELECT min(date), repo_id FROM commits WHERE login = %s AND date <= %s
                 GROUP BY repo_id) S)+1) RN
                ORDER BY S.date DESC
                '''

        contribution_list = self.execute_select_query(query, (login, date, login, date))
        contribution_dict = OrderedDict(map(lambda x: (x["date"], x["NumberOfContributedRepos"]), contribution_list))
        return contribution_dict

    def get_number_of_releases_of_a_user(self, login, date):
        query = '''
                SELECT
                    date, 
                    (SELECT count(*) FROM releases WHERE date <= R.date AND login = R.login) AS ReleaseCount
                FROM `releases` R
                WHERE login = %s AND date <= %s
                GROUP BY date, login
                ORDER BY date DESC
                '''

        release_list = self.execute_select_query(query, (login, date))
        release_dict = OrderedDict(map(lambda x: (x["date"], x["ReleaseCount"]), release_list))
        return release_dict

    def save_daily_developer_stats(self, login, developer_dict):
        query = '''
                INSERT IGNORE daily_developer_stats (login, date, starred_repo_count, forked_repo_count, 
                contributed_repo_count, commit_count, opened_issue_count, resolved_issue_count, release_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                '''
        developer_stats = [(login, date, stats_dict['star'], stats_dict['fork'], stats_dict['contributed_repo'],
                            stats_dict['commit'], stats_dict['opened_issue'], stats_dict['closed_issue'],
                            stats_dict['release']) for (date, stats_dict) in developer_dict.items()]

        self.__executemany_insert_query(query, developer_stats)

    def get_daily_developer_stats(self, category=None):
        query = '''
                SELECT *
                FROM daily_developer_stats
                '''

        developer_stats = self.execute_select_query(query, data=None)
        return developer_stats

    def update_daily_developer_stats(self, developer_stats):
        query = '''
                UPDATE daily_developer_stats
                SET score = %s
                WHERE id = %s
                '''

        data = list(map(lambda x: (x["score"], x["id"]), developer_stats))
        self.__executemany_insert_query(query, data)

    def get_daily_repo_stats_by_day(self, repo_id):
        query = '''
                SELECT *
                FROM daily_repo_stats
                WHERE repo_id = %s AND year(date) >= 2013
                ORDER BY date ASC
                '''

        daily_repo_stats = self.execute_select_query(query, repo_id)
        return daily_repo_stats

    def get_daily_repo_stats_by_week(self, repo_id):
        query = '''
                SELECT week(date) as week,
                       month(date) as month,
                       year(date) as year,
                       repo_id,
                       SUM(commit_count) as CommitSum,
                       SUM(weighted_commit_count) as WeightedCommitSum,
                       SUM(opened_issue_count) as OpenedIssueSum,
                       SUM(weighted_opened_count) as WeightedOpenedIssueSum,
                       SUM(closed_issue_count) as ClosedIssueSum,
                       SUM(weighted_closed_issue_count) as WeightedClosedIssueSum,
                       SUM(release_count) as ReleaseSum,
                       SUM(weighted_release_count) as WeightedReleaseSum,
                       SUM(star_count) as StarSum,
                       SUM(weighted_star_count) as WeightedStarSum,
                       SUM(fork_count) as ForkSum,
                       SUM(weighted_fork_count) as WeightedForkSum
                FROM daily_repo_stats
                WHERE repo_id = %s AND year(date) >= 2013
                GROUP BY week, month, year, repo_id
                ORDER BY year ASC, month ASC, week ASC
                '''

        weekly_repo_stats = self.execute_select_query(query, repo_id)
        return weekly_repo_stats

