import datetime
from collections import defaultdict
from statistics import mean

import requests

from github_trends import secret_config
from github_trends.services.database_service import DatabaseService


class IssueFetcher:
    def __init__(self, context):
        self.db_service = DatabaseService()
        self.last_commit_cursor = None
        self.context = context
        self.api_url = "https://api.github.com/graphql"
        token = secret_config["github-api"]["tokens"][0]
        self.headers = {'Authorization': 'token ' + token}

    def fetch_and_save_issues_of_repo(self, owner, name):
        print("[" + str(datetime.datetime.now()) + "]: Calculating daily issues of repo " +
              owner + "/" + name + " started.")

        commit_list, last_cursor = self.__fetch_issues_of_repo(owner, name)
        parsed_issues = self.__parse_raw_issues(commit_list)
        self.context.AddIssueList(owner + "/" + name, parsed_issues)
        date_issue_dict, daily_user_counts = self.__calculate_daily_stats(parsed_issues)

        self.db_service.save_daily_issues_of_repo(owner, name, date_issue_dict, daily_user_counts)

        print("[" + str(datetime.datetime.now()) + "]: Calculating daily issues of repo " +
              owner + "/" + name + " ended.")

    def __fetch_issues_of_repo(self, owner, name):
        query = '''
                 query ($owner: String!, $name: String!, $after: String) {
                  repository(owner: $owner, name: $name) {
                    issues(first: 100, after: $after, orderBy: {field: CREATED_AT, direction: ASC}) {
                      edges {
                        cursor
                        node {
                          id
                          createdAt
                          updatedAt
                          closed
                          state
                          author {
                            login
                          }
                          timeline(last: 100) {
                            totalCount
                            edges {
                              node() {
                                ... on ClosedEvent {
                                  __typename
                                  createdAt
                                  actor {
                                    login
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                      pageInfo {
                        startCursor
                        hasPreviousPage
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
        issue_list = []
        last_cursor = None

        while True:
            try:
                r = requests.post(self.api_url, headers=self.headers, json={'query': query, 'variables': variables})
                result_json = r.json()
                field_dict = result_json["data"]["repository"]["issues"]

                after_cursor = field_dict["pageInfo"]["endCursor"]
                variables["after"] = after_cursor

                issue_list.extend(result_json["data"]["repository"]["issues"]["edges"])

                if not field_dict["pageInfo"]["hasNextPage"]:
                    last_cursor = after_cursor
                    break
            except:
                print('Error in getting graphql')
                pass

        return issue_list, last_cursor

    @staticmethod
    def __parse_raw_issues(issue_list):
        parsed_issues = []

        for issue in issue_list:
            try:
                reporter = None

                opened_date = datetime.datetime.strptime(issue['node']['createdAt'], '%Y-%m-%dT%H:%M:%SZ')
                closed_date = None
                resolution_seconds = None  # Duration between opening and closing
                if issue['node']['author'] is not None:
                    reporter = issue['node']['author']['login']  # login /username of the user who reports the issue

                issue_timeline = issue['node']['timeline']['edges']
                resolver = None  # login /username of the user who resolves the issue
                for event in issue_timeline:
                    if '__typename' not in event['node']:
                        break
                    event_type = event['node']['__typename']
                    if event_type == "ClosedEvent":
                        closed_date = datetime.datetime.strptime(event['node']['createdAt'], '%Y-%m-%dT%H:%M:%SZ')
                        if event['node']['actor'] is not None:
                            resolver = event['node']['actor']['login']
                        break

                if closed_date is not None:
                    resolution_seconds = (closed_date - opened_date).seconds

                parsed_issues.append({
                    'opened_date': opened_date,
                    'closed_date': closed_date,
                    'resolution_duration': resolution_seconds,
                    'reporter': reporter,
                    'resolver': resolver
                })
            except:
                print('Error in parsing issues')
                pass

        return parsed_issues

    @staticmethod
    def __calculate_daily_stats(parsed_issues):

        # {date : {<"opened"|"closed"|"avg_duration"> : int}}
        daily_issue_counts = defaultdict(lambda: defaultdict(int))

        # {date : {<user_login> : {<"opened"|"closed">:int}}}
        daily_user_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        daily_resolution_times = defaultdict(list)

        for issue in parsed_issues:
            opened_date = issue['opened_date'].date()
            closed_date = issue['closed_date'].date() if issue['closed_date'] is not None else None

            reporter = issue['reporter']
            resolver = issue['resolver']

            daily_issue_counts[opened_date]['opened'] += 1
            if reporter is not None:
                daily_user_counts[opened_date][reporter]['opened'] += 1

            if closed_date is not None:
                daily_issue_counts[opened_date]['closed'] += 1
                daily_resolution_times[closed_date].append(issue['resolution_duration'])
                if resolver is not None:
                    daily_user_counts[opened_date][reporter]['closed'] += 1

        for date, date_list in daily_resolution_times.items():
            avg_resolution_time = mean(date_list)
            daily_issue_counts[date]['avg_duration'] = avg_resolution_time

        return daily_issue_counts, daily_user_counts
