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

    def __execute_select_query(self, query, data):
        conn = self.__get_connection()
        dict_cursor = conn.cursor(pymysql.cursors.DictCursor)

        dict_cursor.execute(query, data)
        values = dict_cursor.fetchall()
        dict_cursor.close()
        return values

    def get_repo_id_by_name_and_owner(self, owner,name):
        query = '''Select id from repos
                   WHERE owner = %s and name = %s
                '''

        repo_id_list = self.__execute_select_query(query, (owner, name))
        return repo_id_list[0]["id"]

    def save_daily_commits_of_repo(self, owner, name, date_commit_dict):
        repo_id = self.get_repo_id_by_name_and_owner(owner, name)
        commit_data = [(repo_id, k, v) for (k, v) in date_commit_dict.items()]

        query = ''' INSERT INTO daily_repo_commits (repo_id, date, commit_count) 
                    VALUES (%s, %s, %s ) '''

        self.__executemany_insert_query(query, commit_data)

    def save_daily_issues_of_repo(self, owner, name, date_issue_dict, date_user_dict):
        repo_id = self.get_repo_id_by_name_and_owner(owner, name)

        issue_data = [(repo_id, date, issue["opened"], issue["closed"], issue["avg_duration"])
                      for (date, issue) in date_issue_dict.items()]
        date_issue_query = ''' INSERT INTO daily_repo_issues (repo_id, date, opened_count, closed_count, avg_resolution_sec) 
                    VALUES ( %s, %s, %s, %s, %s) '''
        self.__executemany_insert_query(date_issue_query, issue_data)

        user_data = [(repo_id, login, date, user["opened"], user["closed"])
                     for (date, user_dict) in date_user_dict.items()
                     for (login, user) in user_dict.items()
                     if login is not None]
        date_user_query = ''' INSERT INTO daily_repo_issue_activities (repo_id, login, date, open_count, closed_count) 
                    VALUES ( %s, %s, %s, %s, %s) '''
        self.__executemany_insert_query(date_user_query, user_data)

    def save_daily_stars_of_repo(self, owner, name, date_star_dict):
        repo_id = self.get_repo_id_by_name_and_owner(owner, name)
        star_data = [(repo_id, k, v) for (k, v) in date_star_dict.items()]

        query = ''' INSERT INTO daily_repo_stars (repo_id, date, star_count) 
                    VALUES ( %s, %s, %s) '''

        self.__executemany_insert_query(query, star_data)

    def save_daily_forks_of_repo(self, owner, name, date_fork_dict):
        repo_id = self.get_repo_id_by_name_and_owner(owner, name)
        fork_data = [(repo_id, k, v) for (k, v) in date_fork_dict.items()]

        query = ''' INSERT INTO daily_repo_forks (repo_id, date, fork_count) 
                    VALUES (%s, %s, %s ) '''

        self.__executemany_insert_query(query, fork_data)

    def save_daily_contributions_of_repo(self, owner, name, date_user_contribution_dict):
        repo_id = self.get_repo_id_by_name_and_owner(owner, name)
        contribution_data = [(repo_id, u, d, c) for d, v in date_user_contribution_dict.items() for u, c in v.items()]

        query = ''' INSERT INTO daily_repo_contributions (repo_id, login, date, commit_count) 
                    VALUES (%s, %s, %s, %s ) '''

        try:
            self.__executemany_insert_query(query, contribution_data)
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

    def save_developer_stats(self, user_stats_dict):
        query = ''' INSERT INTO daily_developer_stats(login, date, starred_repo_count, forked_repo_count,
                    contributed_repo_count, commit_count, opened_issue_count, resolved_issue_count)
                    VALUES( %s, %s, %s, %s, %s, %s, %s, %s)'''

        user_stats = tuple(user_stats_dict.values())

        self.__insert_query(query, user_stats)

    def get_developers(self):
        query = ''' SELECT DISTINCT login FROM daily_repo_contributions '''

        developers = self.__execute_select_query(query, None)

        return developers