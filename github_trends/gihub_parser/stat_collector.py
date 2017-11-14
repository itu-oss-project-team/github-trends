from datetime import datetime, date
from dateutil.rrule import rrule, MONTHLY

from github_trends import category_repos
from github_trends.gihub_parser.commit_fetcher import CommitFetcher
from github_trends.gihub_parser.issue_fetcher import IssueFetcher
from github_trends.gihub_parser.star_and_fork_fetcher import StarAndForkFetcher
from github_trends.gihub_parser.user_analyzer import UserFetcher
from github_trends.gihub_parser.release_fetcher import ReleaseFetcher
from github_trends.services.context import Context
from github_trends.services.database_service import DatabaseService

context = Context()
db_service = DatabaseService()

class StatCollector:
    def __init__(self):
        self.context = context
        self.commit_fetcher = CommitFetcher(self.context)
        self.issue_fetcher = IssueFetcher(self.context)
        self.star_and_fork_fetcher = StarAndForkFetcher(self.context)
        self.release_fetcher = ReleaseFetcher()

    def collect_repo_stats(self, repo_list):
        for repo in repo_list:
            self.release_fetcher.fetch_and_save_releases_of_repo(repo["owner"], repo["name"])
            #self.commit_fetcher.fetch_and_save_commits_of_repo(repo["owner"], repo["name"])
            #self.issue_fetcher.fetch_and_save_issues_of_repo(repo["owner"], repo["name"])
            #self.star_and_fork_fetcher.fetch_and_save_stars_of_repo(repo["owner"], repo["name"])
            #self.star_and_fork_fetcher.fetch_and_save_forks_of_repo(repo["owner"], repo["name"])
        return


if __name__ == '__main__':
    sc = StatCollector()
    uf = UserFetcher(context)
    for category in category_repos.values():
        sc.collect_repo_stats(category)
        '''
        developers = db_service.get_developers()
        for developer in developers:
            start_date = datetime.strptime("2014-01-01", "%Y-%m-%d").date()
            end_date = datetime.strptime("2016-12-31", "%Y-%m-%d").date()
            for dt in rrule(MONTHLY, dtstart=start_date, until=end_date):
                uf.AnalyzeUser(developer["login"], dt.date(), category)
        '''
