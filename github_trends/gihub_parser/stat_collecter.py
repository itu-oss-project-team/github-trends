from github_trends.gihub_parser.commit_fetcher import CommitFetcher
from github_trends.gihub_parser.issue_fetcher import IssueFetcher
from github_trends.gihub_parser.star_and_fork_fetcher import StarAndForkFetcher


class StatCollecter:
    def __init__(self):
        self.commit_fetcher = CommitFetcher()
        self.issue_fetcher = IssueFetcher()
        self.star_and_fork_fetcher = StarAndForkFetcher()

    def collect_repo_stats(self, repo_list):
        for repo in repo_list:
            self.commit_fetcher.fetch_and_save_commits_of_repo(repo["owner"], repo["name"])
            self.issue_fetcher.fetch_and_save_commits_of_repo(repo["owner"], repo["name"])
            self.star_and_fork_fetcher.fetch_and_save_stars_of_repo(repo["owner"], repo["name"])
            self.star_and_fork_fetcher.fetch_and_save_forks_of_repo(repo["owner"], repo["name"])
        return


if __name__ == '__main__':
    sc = StatCollecter()
    repo_list = [{"owner": "d3", "name": "d3"}]
    sc.collect_repo_stats(repo_list)
