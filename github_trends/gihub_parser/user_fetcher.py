from datetime import datetime
from github_trends.services.database_service import DatabaseService

class UserFetcher:
    def __init__(self, context):
        self.context = context
        self.__db_service = DatabaseService()

    def AnalyzeUser(self, user_login, date, category):
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

            try:
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

            except Exception as e:
                print(e)

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




