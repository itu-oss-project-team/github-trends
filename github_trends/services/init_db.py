import os

import pymysql

from github_trends import category_repos
from github_trends import secret_config


def main():
    mysql_config = secret_config['mysql']
    conn = pymysql.connect(host=mysql_config['host'], port=mysql_config['port'], db=mysql_config['db'],
                           user=mysql_config['user'],
                           passwd=mysql_config['passwd'])

    msg = "CAUTION! This will drop all tables and probably going to cause loss of data. Are you sure want to continue?"
    approved = input("%s (y/N) " % msg).lower() == 'y'

    if not approved:
        print("-->Canceled")

        return

    print("-->Dropping tables from \"" + mysql_config['db'] + "\"...")
    clear_db(conn)
    print("-->Tables dropped")
    print("-->Creating tables on \"" + mysql_config['db'] + "\"...")
    init_db(conn)
    print("-->Tables created.")
    print("-->Inserting categories and repositories on \"" + mysql_config['db'] + "\"...")
    init_categories_and_repos(conn)
    print("-->Categories and repositories inserted.")


def clear_db(conn):
    cursor = conn.cursor()
    cursor.execute("""
            SET foreign_key_checks = 0;
            DROP TABLE IF EXISTS
            `categories`, `category_repos`, `daily_repo_commits`, `daily_repo_contributions`, `daily_repo_forks`, 
            `daily_repo_issue_activities`, `daily_repo_issues`,`daily_repo_releases`, `daily_repo_stars`, 
            `daily_stackoverflow_questions`, `repos`;
            SET foreign_key_checks = 1;
    """)

    conn.commit()


def init_db(conn):
    with open(os.path.join(os.path.dirname(__file__), 'db_create_query.sql'), 'r') as db_creation_file:
        cursor = conn.cursor()
        db_creation_query = db_creation_file.read()

        cursor.execute(db_creation_query)

        conn.commit()


def init_categories_and_repos(conn):
    cursor = conn.cursor()

    for category, repos in category_repos.items():
        cursor.execute('''INSERT INTO categories (name) VALUES ( %s ) ''', category)
        category_id = cursor.lastrowid

        for repo in repos:
            cursor.execute(''' INSERT IGNORE repos (full_name, name, owner, so_tag) 
                    VALUES ( %s, %s, %s, %s ) ''', (repo['full_name'], repo['name'], repo['owner'], repo['so_tag']))
            repo_id = cursor.lastrowid
            conn.commit()

            if repo_id == 0:
                cursor.execute('''SELECT id FROM repos WHERE owner = %s AND name = %s''',
                               (repo['owner'], repo['name']))
                repo_id = cursor.fetchall()[0][0]

            cursor.execute(''' INSERT INTO category_repos (category_id, repo_id) 
                    VALUES ( %s, %s) ''', (category_id, repo_id))

    conn.commit()
    cursor.close()


main()
