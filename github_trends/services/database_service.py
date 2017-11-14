import pymysql
from collections import OrderedDict

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
        release_data = [(repo_id, k, v) for (k,v) in date_release_dict.items()]

        query = ''' INSERT IGNORE daily_repo_releases (repo_id, date, release_count)
                    VALUES (%s, %s, %s) '''

        try:
            self.__executemany_insert_query(query, release_data)
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
        query = ''' INSERT IGNORE daily_developer_stats(login, date, starred_repo_count, forked_repo_count,
                    contributed_repo_count, commit_count, opened_issue_count, resolved_issue_count)
                    VALUES( %s, %s, %s, %s, %s, %s, %s, %s)'''

        user_stats = tuple(user_stats_dict.values())

        self.__insert_query(query, user_stats)

    def get_developers(self):
        query = ''' SELECT DISTINCT login FROM daily_repo_contributions '''

        developers = self.__execute_select_query(query, None)

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
            x["reporter"],  x["resolver"], repo_id, x["resolution_duration"]), issue_list))

        query = ''' INSERT INTO issues(opened_date, resolved_date, reporter, resolver, repo_id, resolution_duration_sec)
                    VALUES (%s, %s, %s, %s, %s, %s) '''

        self.__executemany_insert_query(query, issue_tuple_list)

    def save_releases(self, owner, name, release_list):
        repo_id = self.get_repo_id_by_name_and_owner(owner, name)

        release_tuple_list = list(map(lambda x: (x["date"], x["login"], repo_id)), release_list)
        query = ''' INSERT INTO releases(date, login, repo_id)
                    VALUES (%s, %s, %s) '''

        self.__executemany_insert_query(query, release_tuple_list)


    def get_cumulative_commits_of_a_user(self, login, date):
        query = '''
            SELECT DISTINCT 
            C.date, 
            (select count(*) from commits innerC where innerC.date <= C.date and innerC.login = C.login) as NumberOfCommits
            FROM `commits` C
            where login = %s and C.date <= %s
            order by C.date desc
                '''

        commit_list = self.__execute_select_query(query, (login, date))
        commit_dict = OrderedDict(map(lambda x: (x["date"], x["NumberOfCommits"]), commit_list))
        return commit_dict

    def get_cumulative_opened_issues_of_a_user(self, login, date):
        query = '''
            SELECT DISTINCT 
            I.opened_date,
            (select count(*) from issues innerI where innerI.opened_date <= I.opened_date and innerI.reporter = I.reporter) as NumberOfIssuesOpened                
            FROM issues I
            where I.reporter = %s and I.opened_date <= %s
            order by I.opened_date desc
                '''

        opened_issue_list = self.__execute_select_query(query, (login, date))
        opened_issue_dict = OrderedDict(map(lambda x: (x["opened_date"], x["NumberOfIssuesOpened"]), opened_issue_list))
        return opened_issue_dict

    def get_cumulative_closed_issues_of_a_user(self, login, date):
        query = '''
            SELECT DISTINCT 
            I.resolved_date,
            (select count(*) from issues innerI where innerI.resolved_date <= I.resolved_date and innerI.resolver = I.resolver) as NumberOfIssuesClosed
            FROM issues I
            where I.resolver = %s and I.resolved_date <= %s
            order by I.resolved_date desc                        
              '''

        opened_issue_list = self.__execute_select_query(query, (login, date))
        opened_issue_dict = OrderedDict(map(lambda x: (x["resolved_date"], x["NumberOfIssuesClosed"]), opened_issue_list))
        return opened_issue_dict

    def get_number_of_starred_repos_of_a_user(self, login, date):
        query = '''
                SELECT
                    date, 
                    count(*) as StarCount
                FROM `stars` S
                WHERE login = %s and date <= %s
                group by date, login
                order by date desc
                '''

        star_list = self.__execute_select_query(query, (login, date))
        star_dict = OrderedDict(map(lambda x: (x["date"], x["StarCount"]), star_list))
        return star_dict

    def get_number_of_forked_repos_of_a_user(self, login, date):
        query = '''
                SELECT 
                    date, 
                    count(*) as ForkCount
                FROM `forks` F
                WHERE login = %s and date <= %s
                group by date, login  
                ORDER BY date DESC
                '''

        fork_list = self.__execute_select_query(query, (login, date))
        fork_dict = OrderedDict(map(lambda x: (x["date"], x["StarCount"]), fork_list))
        return fork_dict

    def get_number_of_contributed_repos_of_a_user(self, login, date):
        query = '''
                SET @rank=(SELECT Count(*) from 
                    (SELECT min(date), repo_id FROM commits WHERE login = %s and date <= %s
                     GROUP BY repo_id) S)+1;
                
                SELECT min(date) as date, @rank:=@rank-1 AS NumberOfContributedRepos FROM commits 
                WHERE login = %s and date <= %s
                GROUP BY repo_id
                '''

        contribution_list = self.__execute_select_query(query, (login, date, login, date))
        contribution_dict = OrderedDict(map(lambda x: (x["date"], x["NumberOfContributedRepos "]), contribution_list))
        return contribution_dict


    


