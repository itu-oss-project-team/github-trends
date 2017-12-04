import pandas as pd
import pymysql

from github_trends import secret_config

mysql_config = secret_config['mysql']
mysql_config = mysql_config
conn = pymysql.connect(host=mysql_config['host'], port=mysql_config['port'], db=mysql_config['db'],
                       user=mysql_config['user'], passwd=mysql_config['passwd'], charset='utf8mb4',
                       use_unicode=True)

get_user_stats_query = '''
    SELECT 
    id,
    starred_repo_count,
    forked_repo_count,
    contributed_repo_count,
    commit_count,
    opened_issue_count,
    resolved_issue_count,
    release_count
    FROM daily_developer_stats
'''

correlations = pd.read_sql(sql=get_user_stats_query, con=conn,
                            index_col="id")

print(correlations.corr('kendall'))