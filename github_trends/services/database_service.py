import pymysql
from github_trends import secret_config


class DatabaseService:
    def __init__(self):
        self.__mysql_config = secret_config['mysql']

    def __executemany_insert_query(self, query, data):
        mysql_config = self.__mysql_config
        conn = pymysql.connect(host=mysql_config['host'], port=mysql_config['port'], db=mysql_config['db'],
                                    user=mysql_config['user'], passwd=mysql_config['passwd'], charset='utf8mb4',
                                    use_unicode=True)

        dict_cursor = conn.cursor(pymysql.cursors.DictCursor)
        dict_cursor.executemany(query, data)
        conn.commit()
        dict_cursor.close()

    def save_daily_commits_of_repo(self, owner, name, date_commit_dict):
        commit_data = [(owner, name, k, v) for (k,v) in date_commit_dict.items()]
        query = ''' INSERT INTO daily_repo_commits (repo_owner, repo_name, date, commit_count) 
                    VALUES ( %s, %s, %s, %s ) '''

        self.__executemany_insert_query(query, commit_data)

    def save_daily_stars_of_repo(self, owner, name, date_star_dict):
        star_data = [(owner, name, k, v) for (k,v) in date_star_dict.items()]
        query = ''' INSERT INTO daily_repo_stars (repo_owner, repo_name, date, star_count) 
                    VALUES ( %s, %s, %s, %s ) '''

        self.__executemany_insert_query(query, star_data)

    def save_daily_forks_of_repo(self, owner, name, date_fork_dict):
        fork_data = [(owner, name, k, v) for (k,v) in date_fork_dict.items()]
        query = ''' INSERT INTO daily_repo_forks (repo_owner, repo_name, date, fork_count) 
                    VALUES ( %s, %s, %s, %s ) '''

        self.__executemany_insert_query(query, fork_data)
