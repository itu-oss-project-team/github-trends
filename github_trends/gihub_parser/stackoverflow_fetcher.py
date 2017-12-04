from collections import defaultdict

from github_trends import category_repos
from github_trends.services.database_service import DatabaseService


class StackOverflowFetcher:
    def __init__(self):
        self.__db_service = DatabaseService()

    def __fetch_stack_overflow_stats(self, so_tags):
        query = '''
        SELECT
          repo_tag,
          date,
          COUNT(*) question_count,
          SUM(view_count) view_sum,
          SUM(favorite_count) favorite_sum,
          SUM(score) score_sum,
          SUM(answer_count) answer_sum
        FROM (
          SELECT
            view_count,
            favorite_count,
            score,
            answer_count,
            (SELECT tag FROM UNNEST({}) tag WHERE tag IN UNNEST(SPLIT(tags, "|")) LIMIT 1) AS repo_tag,
            EXTRACT(DATE FROM creation_date) AS date
          FROM
            `bigquery-public-data.stackoverflow.posts_questions` ) S
        WHERE
          repo_tag IN UNNEST({})
        GROUP BY
          date,
          repo_tag
        '''.format(so_tags, so_tags)

        results = self.__db_service.execute_bigquery_select(query)
        return results

    def fetch_and_save_so_stats_of_repo(self, repos):
        tag_full_name_dict = {repo['so_tag']:repo['full_name'] for repo in repos if repo['so_tag']}
        so_tags = list(tag_full_name_dict.keys())

        raw_stats = self.__fetch_stack_overflow_stats(so_tags)

        repo_stat_dict = defaultdict(list)
        for stat in raw_stats:
            repo_stat_dict[tag_full_name_dict[stat['repo_tag']]].append(stat)

        for full_name, stats in repo_stat_dict.items():
            self.__db_service.save_daily_so_stats_of_repo(full_name, stats)


sof = StackOverflowFetcher()
sof.fetch_and_save_so_stats_of_repo(category_repos['visualization'])

