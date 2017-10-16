import datetime
import requests
import collections

from github_trends import secret_config
from github_trends.services.database_service import DatabaseService


class StarAndForkFetcher:
    def __init__(self):
        self.db_service = DatabaseService()
        self.last_commit_cursor = None
        self.api_url = "https://api.github.com/graphql"
        token = secret_config["github-api"]["tokens"][0]
        self.headers = {'Authorization': 'token ' + token}


    def __fetch_stars_of_repo(self, owner, name):
        query = '''
                query($owner:String!, $name:String!, $after:String){
                  repository(owner: $owner, name: $name) {
                    stargazers(first: 100, orderBy: {field: STARRED_AT, direction: ASC}, after: $after) {
                      edges {
                        starredAt
                        cursor
                        node {
                          login
                        }
                      }
                      totalCount
                      pageInfo {
                        endCursor
                        hasNextPage
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
        stargazer_list = []
        while True:
            r = requests.post(self.api_url, headers=self.headers, json={'query': query, 'variables': variables})
            result_json = r.json()
            field_dict = result_json["data"]["repository"]["stargazers"]
            after_cursor = field_dict["pageInfo"]["endCursor"]
            variables["after"] = after_cursor
            stargazer_list.extend(field_dict["edges"])

            if not field_dict["pageInfo"]["hasNextPage"]:
                last_cursor = after_cursor
                break

        return stargazer_list, last_cursor

    def __fetch_forks_of_repo(self, owner, name):
        query = '''
                query ($owner: String!, $name: String!, $after: String) {
                  repository(owner: $owner, name: $name) {
                    createdAt
                    forks(first:100, orderBy:{field: CREATED_AT, direction:ASC}, after: $after)
                    {
                      edges {
                        cursor
                        node {
                          nameWithOwner
                          createdAt
                        }
                      }
                      pageInfo {
                        hasNextPage
                        endCursor
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
        forks_list = []
        while True:
            r = requests.post(self.api_url, headers=self.headers, json={'query': query, 'variables': variables})
            result_json = r.json()
            field_dict = result_json["data"]["repository"]["forks"]
            after_cursor = field_dict["pageInfo"]["endCursor"]
            variables["after"] = after_cursor
            forks_list.extend(field_dict["edges"])

            if not field_dict["pageInfo"]["hasNextPage"]:
                last_cursor = after_cursor
                break

        return forks_list, last_cursor

    def __calculate_daily_stars(self, star_list):
        date_stars_dict = collections.OrderedDict()

        for edge in star_list:
            date = datetime.datetime.strptime(edge["starredAt"], "%Y-%m-%dT%H:%M:%SZ").date()

            if date not in date_stars_dict:
                date_stars_dict[date] = 1
            else:
                date_stars_dict[date] += 1

        return date_stars_dict

    def __calculate_daily_forks(self, fork_list):
        date_fork_dict = collections.OrderedDict()

        for edge in fork_list:
            date = datetime.datetime.strptime(edge["node"]["createdAt"], "%Y-%m-%dT%H:%M:%SZ").date()

            if date not in date_fork_dict:
                date_fork_dict[date] = 1
            else:
                date_fork_dict[date] += 1

        return date_fork_dict

    def fetch_and_save_stars_of_repo(self, owner, name):
        print("[" + str(datetime.datetime.now()) + "]: Calculating daily stars of repo " +
              owner + '/' + name + " started.")

        star_list, last_cursor = self.__fetch_stars_of_repo(owner, name)
        date_stars_dict = self.__calculate_daily_stars(star_list)

        self.db_service.save_daily_stars_of_repo(owner, name, date_stars_dict)

        print("[" + str(datetime.datetime.now()) + "]: Calculating daily stars of repo " +
              owner + "/" + name + " ended.")

    def fetch_and_save_forks_of_repo(self, owner, name):
        print("[" + str(datetime.datetime.now()) + "]: Calculating daily forks of repo " +
              owner + '/' + name + " started.")

        fork_list, last_cursor = self.__fetch_forks_of_repo(owner, name)
        date_fork_dict = self.__calculate_daily_forks(fork_list)

        self.db_service.save_daily_forks_of_repo(owner, name, date_fork_dict)

        print("[" + str(datetime.datetime.now()) + "]: Calculating daily forks of repo " +
              owner + "/" + name + " ended.")
