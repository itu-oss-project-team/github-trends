import requests
from datetime import datetime
from collections import defaultdict, OrderedDict

from github_trends import secret_config
from github_trends.services.database_service import DatabaseService


class CommitFetcher:
    def __init__(self):
        self.db_service = DatabaseService()
        self.last_commit_cursor = None
        self.api_url = "https://api.github.com/graphql"
        token = secret_config["github-api"]["tokens"][0]
        self.headers = {'Authorization': 'token ' + token}

    def __fetch_commits_of_repo(self, owner, name):
        query = '''
                 query ($owner: String!, $name: String!, $after: String) {
                  repository(owner: $owner, name: $name) {
                    createdAt
                    ref(qualifiedName:"master"){
                      target {
                        ... on Commit {
                          history(first:100, after:$after){
                            edges {
                              cursor
                              node {
                                author {user {login}}
                                committedDate
                              }
                            }
                            pageInfo {hasNextPage,endCursor}
                          }
                        }
                      }
                    }
                  }
                  rateLimit {
                    limit
                    cost
                    remaining
                    resetAt
                  }
                }
                '''

        variables = {"owner": owner, "name": name, "after": None}
        commit_list = []

        while True:
            r = requests.post(self.api_url, headers=self.headers, json={'query': query, 'variables': variables})
            result_json = r.json()
            field_dict = result_json["data"]["repository"]["ref"]["target"]["history"]

            after_cursor = field_dict["pageInfo"]["endCursor"]
            variables["after"] = after_cursor
            commit_list.extend(result_json["data"]["repository"]["ref"]["target"]["history"]["edges"])

            if not field_dict["pageInfo"]["hasNextPage"]:
                last_cursor = after_cursor
                break

        return commit_list, last_cursor

    def __calculate_daily_results(self, commit_list):
        date_commit_dict = OrderedDict()

        for edge in commit_list:
            date = datetime.strptime(edge["node"]["committedDate"], "%Y-%m-%dT%H:%M:%SZ").date()

            if date not in date_commit_dict:
                date_commit_dict[date] = 1
            else:
                date_commit_dict[date] += 1

        return date_commit_dict

    def __get_users_and_contributions(self, commit_list):
        contribution_dict = defaultdict(lambda: defaultdict(int))  # {date:<user_login>:int}}
        for edge in commit_list:
            try:
                user_login = edge["node"]["author"]["user"]["login"]
                date = datetime.strptime(edge["node"]["committedDate"], "%Y-%m-%dT%H:%M:%SZ").date()

                contribution_dict[date][user_login] += 1
            except:
                print('Error in getting commit date and contribution information!')
                pass

        return contribution_dict

    def fetch_and_save_commits_of_repo(self, owner, name):
        print("[" + str(datetime.now()) + "]: Calculating daily commits of repo " +
              owner + "/" + name + " started.")

        commit_list, last_cursor = self.__fetch_commits_of_repo(owner, name)
        commit_list = list(filter(lambda x: x["node"]["author"]["user"] is not None, commit_list))

        date_commit_dict = self.__calculate_daily_results(commit_list)
        date_user_contribution_dict = self.__get_users_and_contributions(commit_list)

        commit_list = list(map(lambda x:
                        {"login": x["node"]["author"]["user"]["login"] if x["node"]["author"]["user"] is not None else None,
                         "date": datetime.strptime(x["node"]["committedDate"], "%Y-%m-%dT%H:%M:%SZ").date()}, commit_list))

        self.db_service.save_commits(owner, name, commit_list)
        self.db_service.save_daily_commits_of_repo(owner, name, date_commit_dict)
        self.db_service.save_daily_contributions_of_repo(owner, name, date_user_contribution_dict)

        print("[" + str(datetime.now()) + "]: Calculating daily commits of repo " +
              owner + "/" + name + " ended.")

