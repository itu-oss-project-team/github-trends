-- Created by Vertabelo (http://vertabelo.com)
-- Last modification date: 2017-10-24 21:17:11.287

-- tables
-- Table: categories
CREATE TABLE categories (
    id int NOT NULL AUTO_INCREMENT,
    name varchar(255) NOT NULL,
    CONSTRAINT categories_pk PRIMARY KEY (id)
);

-- Table: category_repos
CREATE TABLE category_repos (
    id int NOT NULL AUTO_INCREMENT,
    repo_id int NOT NULL,
    category_id int NOT NULL,
    CONSTRAINT category_repos_pk PRIMARY KEY (id)
);

-- Table: daily_repo_commits
CREATE TABLE daily_repo_commits (
    id int NOT NULL AUTO_INCREMENT,
    repo_id int NOT NULL,
    date date NOT NULL,
    commit_count int NOT NULL,
    CONSTRAINT daily_repo_commits_pk PRIMARY KEY (id)
);

-- Table: daily_repo_contributions
CREATE TABLE daily_repo_contributions (
    id int NOT NULL AUTO_INCREMENT,
    repo_id int NOT NULL,
    login varchar(255) NOT NULL,
    date date NOT NULL,
    commit_count int NOT NULL DEFAULT 0,
    UNIQUE INDEX unique_daily_contribution (repo_id,login,date),
    CONSTRAINT daily_repo_contributions_pk PRIMARY KEY (id)
);

-- Table: daily_repo_forks
CREATE TABLE daily_repo_forks (
    id int NOT NULL AUTO_INCREMENT,
    repo_id int NOT NULL,
    date date NOT NULL,
    fork_count int NOT NULL,
    CONSTRAINT daily_repo_forks_pk PRIMARY KEY (id)
);

-- Table: daily_repo_issue_activities
CREATE TABLE daily_repo_issue_activities (
    id int NOT NULL AUTO_INCREMENT,
    repo_id int NOT NULL,
    login varchar(255) NOT NULL,
    date date NOT NULL,
    open_count int NOT NULL DEFAULT 0,
    closed_count int NOT NULL DEFAULT 0,
    comment_count int NULL,
    CONSTRAINT daily_repo_issue_activities_pk PRIMARY KEY (id)
);

-- Table: daily_repo_issues
CREATE TABLE daily_repo_issues (
    id int NOT NULL AUTO_INCREMENT,
    repo_id int NOT NULL,
    date date NOT NULL,
    opened_count int NULL COMMENT 'Number of issues opened that day',
    closed_count int NULL COMMENT 'Number of issues closed that day',
    comment_count int NULL COMMENT 'Number of comments made on issues that day',
    avg_resolution_sec int NULL,
    CONSTRAINT daily_repo_issues_pk PRIMARY KEY (id)
);

-- Table: daily_repo_releases
CREATE TABLE daily_repo_releases (
    id int NOT NULL AUTO_INCREMENT,
    repo_id int NOT NULL,
    date date NOT NULL,
    release_count int NOT NULL,
    UNIQUE INDEX repo_date_ak (date,repo_id),
    CONSTRAINT daily_repo_releases_pk PRIMARY KEY (id)
);

-- Table: daily_repo_stars
CREATE TABLE daily_repo_stars (
    id int NOT NULL AUTO_INCREMENT,
    repo_id int NOT NULL,
    date date NOT NULL,
    star_count int NOT NULL,
    CONSTRAINT daily_repo_stars_pk PRIMARY KEY (id)
);

-- Table: daily_stackoverflow_questions
CREATE TABLE daily_stackoverflow_questions (
    id int NOT NULL AUTO_INCREMENT,
    repo_id int NOT NULL,
    question_count int NOT NULL,
    CONSTRAINT daily_stackoverflow_questions_pk PRIMARY KEY (id)
);

-- Table: repos
CREATE TABLE repos (
    id int NOT NULL AUTO_INCREMENT,
    full_name varchar(255) NOT NULL COMMENT 'This is the formal notation for repo names. Like "torvalds/linux"',
    owner varchar(255) NOT NULL COMMENT 'This is the lhs of full name. (For "torvalds/linux" this field is "torvalds")',
    name varchar(255) NOT NULL COMMENT 'This is the rhs of full name. (For "torvalds/linux" this field is "linux")',
    github_id int NULL COMMENT 'ID of this repo in GitHub resources',
    ghtorrent_id int NULL COMMENT 'ID of this repo in GHTorrent resources',
    so_tag varchar(255) NULL COMMENT 'Dedicated StackOverflow tag for this repo',
    UNIQUE INDEX unique_full_name (full_name),
    CONSTRAINT repos_pk PRIMARY KEY (id)
);

-- foreign keys
-- Reference: Copy_of_daily_repo_issues_repos (table: daily_repo_commits)
ALTER TABLE daily_repo_commits ADD CONSTRAINT Copy_of_daily_repo_issues_repos FOREIGN KEY Copy_of_daily_repo_issues_repos (repo_id)
    REFERENCES repos (id);

-- Reference: Copy_of_daily_repo_stars_repos (table: daily_repo_contributions)
ALTER TABLE daily_repo_contributions ADD CONSTRAINT Copy_of_daily_repo_stars_repos FOREIGN KEY Copy_of_daily_repo_stars_repos (repo_id)
    REFERENCES repos (id);

-- Reference: daily_repo_forks_repos (table: daily_repo_forks)
ALTER TABLE daily_repo_forks ADD CONSTRAINT daily_repo_forks_repos FOREIGN KEY daily_repo_forks_repos (repo_id)
    REFERENCES repos (id);

-- Reference: daily_repo_issue_activities_repos (table: daily_repo_issue_activities)
ALTER TABLE daily_repo_issue_activities ADD CONSTRAINT daily_repo_issue_activities_repos FOREIGN KEY daily_repo_issue_activities_repos (repo_id)
    REFERENCES repos (id);

-- Reference: daily_repo_issues_repos (table: daily_repo_issues)
ALTER TABLE daily_repo_issues ADD CONSTRAINT daily_repo_issues_repos FOREIGN KEY daily_repo_issues_repos (repo_id)
    REFERENCES repos (id);

-- Reference: daily_repo_releases_repos (table: daily_repo_releases)
ALTER TABLE daily_repo_releases ADD CONSTRAINT daily_repo_releases_repos FOREIGN KEY daily_repo_releases_repos (repo_id)
    REFERENCES repos (id);

-- Reference: daily_repo_stars_repos (table: daily_repo_stars)
ALTER TABLE daily_repo_stars ADD CONSTRAINT daily_repo_stars_repos FOREIGN KEY daily_repo_stars_repos (repo_id)
    REFERENCES repos (id);

-- Reference: daily_stackoverflow_questions_repos (table: daily_stackoverflow_questions)
ALTER TABLE daily_stackoverflow_questions ADD CONSTRAINT daily_stackoverflow_questions_repos FOREIGN KEY daily_stackoverflow_questions_repos (repo_id)
    REFERENCES repos (id);

-- Reference: list_repos_lists (table: category_repos)
ALTER TABLE category_repos ADD CONSTRAINT list_repos_lists FOREIGN KEY list_repos_lists (category_id)
    REFERENCES categories (id);

-- Reference: list_repos_repos (table: category_repos)
ALTER TABLE category_repos ADD CONSTRAINT list_repos_repos FOREIGN KEY list_repos_repos (repo_id)
    REFERENCES repos (id);

-- End of file.
