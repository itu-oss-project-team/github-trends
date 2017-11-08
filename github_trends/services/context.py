class Context:
    def __init__(self):
        self.star_list = {}
        self.fork_list = {}
        self.commit_list = {}
        self.issue_list = {}

    def GetStarListByRepoFullName(self, repo_full_name):
        return self.star_list.get(repo_full_name, None)

    def GetForkListByRepoFullName(self, repo_full_name):
        return self.fork_list.get(repo_full_name, None)

    def GetCommitListByRepoFullName(self, repo_full_name):
        return self.commit_list.get(repo_full_name, None)

    def GetIssueListByRepoFullName(self, repo_full_name):
        return self.issue_list.get(repo_full_name, None)

    def AddStarList(self, repo_full_name, star_list):
        self.star_list[repo_full_name] = star_list

    def AddForkList(self, repo_full_name, fork_list):
        self.fork_list[repo_full_name] = fork_list

    def AddCommitList(self, repo_full_name, commit_list):
        self.commit_list[repo_full_name] = commit_list

    def AddIssueList(self, repo_full_name, issue_list):
        self.issue_list[repo_full_name] = issue_list