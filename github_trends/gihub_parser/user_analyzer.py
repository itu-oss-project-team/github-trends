from datetime import datetime
from collections import OrderedDict, defaultdict
from pprint import pprint

from github_trends.services.database_service import DatabaseService

db_service = DatabaseService()

class UserFetcher:
    def __init__(self):
        self.db_service = db_service

    def AnalyzeUser(self, user_login, date, category=None):

        commit_counts = self.db_service.get_cumulative_commits_of_a_user(user_login, date)
        opened_issue_counts = self.db_service.get_cumulative_opened_issues_of_a_user(user_login, date)
        closed_issue_counts = self.db_service.get_cumulative_closed_issues_of_a_user(user_login, date)
        star_counts = self.db_service.get_number_of_starred_repos_of_a_user(user_login, date)
        fork_counts = self.db_service.get_number_of_forked_repos_of_a_user(user_login, date)
        contributed_repo_counts = self.db_service.get_number_of_contributed_repos_of_a_user(user_login, date)
        release_counts = self.db_service.get_number_of_releases_of_a_user(user_login, date)

        counts_dicts = {
            'commit': commit_counts,
            'opened_issue': opened_issue_counts,
            'closed_issue': closed_issue_counts,
            'star': star_counts,
            'fork': fork_counts,
            'contributed_repo': contributed_repo_counts,
            'release': release_counts
        }

        ##########

        dates_set = sorted(set().union([date_key for counts in counts_dicts.values() for date_key in counts]))
        total_dict = defaultdict(lambda: defaultdict(int))

        for data_key, counts_dict in counts_dicts.items():
            if counts_dict:
                last_date, current_count = counts_dict.popitem(last=True)
            else:
                last_date = datetime(1900, 1, 1).date()
                current_count = 0

            last_count = 0

            for date in dates_set:
                if date >= last_date:
                    last_count = current_count
                    if counts_dict:
                        last_date, current_count = counts_dict.popitem()
                total_dict[date][data_key] += last_count

        self.db_service.save_daily_developer_stats(user_login, total_dict)

    def AnalyzeUserOld(self, user_login, date, category):
        print('User ', user_login, ' Date: ', str(date))

        numberOfReposStarred = 0
        numberOfReposContributed = 0
        numberOfReposForked = 0
        numberOfContributions = 0
        numberOfIssuesOpened = 0
        numberOfIssuesResolved = 0

        for repo in category:
            star_list = self.context.GetStarListByRepoFullName(repo["full_name"])
            fork_list = self.context.GetForkListByRepoFullName(repo["full_name"])
            commit_list = self.context.GetCommitListByRepoFullName(repo["full_name"])
            issue_list = self.context.GetIssueListByRepoFullName(repo["full_name"])

            star_list = list(filter(lambda x:
                        x["node"]["login"] == user_login and
                        datetime.strptime(x["starredAt"], "%Y-%m-%dT%H:%M:%SZ").date() < date, star_list))

            fork_list = list(filter(lambda x:
                        x["node"]["owner"]["login"] == user_login and
                        datetime.strptime(x["node"]["createdAt"],"%Y-%m-%dT%H:%M:%SZ").date() < date, fork_list))

            commit_list = list(filter(lambda x: x["node"]["author"]["user"] is not None, commit_list))
            commit_list = list(filter(lambda x:
                        x["node"]["author"]["user"]["login"] == user_login and
                        datetime.strptime(x["node"]["committedDate"],"%Y-%m-%dT%H:%M:%SZ").date() < date, commit_list))

            opened_issue_list = list(filter(lambda x:
                        x["reporter"] == user_login and ("opened_date" in x and x["opened_date"].date() < date), issue_list))

            closed_issue_list = list(filter(lambda x:
                        x["resolver"] == user_login and ("closed_date" in x and x["closed_date"].date() < date), issue_list))

            if len(star_list) > 0:
                numberOfReposStarred += 1

            if len(fork_list) > 0:
                numberOfReposForked += 1

            if len(commit_list) > 0:
                numberOfReposContributed += 1
                numberOfContributions += len(commit_list)

            if len(opened_issue_list) > 0:
                numberOfIssuesOpened += len(opened_issue_list)

            if len(closed_issue_list) > 0:
                numberOfIssuesResolved += len(closed_issue_list)

        user_stats = {}
        user_stats["login"] = user_login
        user_stats["date"] = date
        user_stats["starred_repo_count"] = numberOfReposStarred
        user_stats["forked_repo_count"] = numberOfReposForked
        user_stats["contributed_repo_count"] = numberOfReposContributed
        user_stats["commit_count"] = numberOfContributions
        user_stats["opened_issue_count"] = numberOfIssuesOpened
        user_stats["resolved_issue_count"] = numberOfIssuesResolved

        try:
            self.__db_service.save_developer_stats(user_stats)
        except:
            pass


if __name__ == "__main__":
    uf = UserFetcher()
    developers = db_service.get_developers()
    for developer in developers:
        print("[" + str(datetime.now()) + "]: " + developer["login"])
        uf.AnalyzeUser(developer["login"], datetime(2016, 12, 31).date())

