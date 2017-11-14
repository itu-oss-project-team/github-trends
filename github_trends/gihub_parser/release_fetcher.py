import requests
from datetime import datetime
from collections import defaultdict, OrderedDict

from github_trends import secret_config
from github_trends.services.database_service import DatabaseService


class ReleaseFetcher:
    def __init__(self):
        self.db_service = DatabaseService()
        self.api_url = "https://api.github.com/graphql"
        token = secret_config["github-api"]["tokens"][0]
        self.headers = {'Authorization': 'token ' + token}

    def __fetch_releases_of_repo(self, owner, name):
        query = '''
                query ($owner: String!, $name: String!, $after: String) {
                  repository(owner: $owner, name: $name) {
                    releases(first: 100, after: $after) {
                      totalCount
                      edges {
                        cursor
                        node {
                          author {
                            login
                          }
                          createdAt
                          isDraft
                          isPrerelease
                          publishedAt
                          updatedAt
                        }
                      }
                      pageInfo {
                        hasNextPage
                        hasPreviousPage
                        endCursor
                        startCursor
                      }
                    }
                  }
                }
                '''

        variables = {"owner": owner, "name": name, "after": None}
        release_list = []

        while True:
            r = requests.post(self.api_url, headers=self.headers, json={'query': query, 'variables': variables})
            result_json = r.json()
            field_dict = result_json["data"]["repository"]["releases"]

            after_cursor = field_dict["pageInfo"]["endCursor"]
            variables["after"] = after_cursor
            release_list.extend(result_json["data"]["repository"]["releases"]["edges"])

            if not field_dict["pageInfo"]["hasNextPage"]:
                last_cursor = after_cursor
                break

        return release_list, last_cursor

    def __calculate_daily_results(self, release_list):
        date_release_dict = defaultdict(int)

        for edge in release_list:
            date = datetime.strptime(edge["node"]["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").date()
            date_release_dict[date] += 1

        return date_release_dict

    def fetch_and_save_releases_of_repo(self, owner, name):
        print("[" + str(datetime.now()) + "]: Calculating daily releases of repo " +
              owner + "/" + name + " started.")

        release_list, last_cursor = self.__fetch_releases_of_repo(owner, name)
        date_release_dict = self.__calculate_daily_results(release_list)

        release_list = list(map(lambda x: {"login": x["node"]["author"]["login"],
                                "date": datetime.strptime(x["node"]["createdAt"], "%Y-%m-%dT%H:%M:%SZ").date()},
                                release_list))

        self.db_service.save_daily_releases_of_repo(owner, name, date_release_dict)
        self.db_service.save_releases(owner, name, release_list)

        print("[" + str(datetime.now()) + "]: Calculating daily releases of repo " +
              owner + "/" + name + " ended.")

