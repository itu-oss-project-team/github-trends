-- Created by Vertabelo (http://vertabelo.com)
-- Last modification date: 2018-06-21 11:48:38.05

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

-- Table: commits
CREATE TABLE commits (
    id int NOT NULL AUTO_INCREMENT,
    date date NOT NULL,
    login varchar(255) NULL,
    repo_id int NOT NULL,
    CONSTRAINT commits_pk PRIMARY KEY (id)
);

-- Table: daily_developer_stats
CREATE TABLE daily_developer_stats (
    id int NOT NULL AUTO_INCREMENT,
    login varchar(255) NOT NULL,
    category_id int NOT NULL,
    date date NOT NULL,
    starred_repo_count int NULL DEFAULT 0,
    forked_repo_column int NULL DEFAULT 0,
    contributed_repo_count int NULL DEFAULT 0,
    commit_count int NULL DEFAULT 0,
    opened_issue_count int NULL DEFAULT 0,
    resolved_issue_count int NULL DEFAULT 0,
    release_count int NULL DEFAULT 0,
    UNIQUE INDEX unique_daily_developer_stats (login,date),
    CONSTRAINT daily_developer_stats_pk PRIMARY KEY (id)
);

-- Table: daily_repo_commits
CREATE TABLE daily_repo_commits (
    id int NOT NULL AUTO_INCREMENT,
    repo_id int NOT NULL,
    date date NOT NULL,
    commit_count int NOT NULL DEFAULT 0,
    UNIQUE INDEX daily_repo_commits_ak_1 (repo_id,date),
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
    fork_count int NOT NULL DEFAULT 0,
    UNIQUE INDEX daily_repo_forks_ak_1 (repo_id,date),
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
    UNIQUE INDEX daily_repo_issue_activities_ak_1 (repo_id,date,login),
    CONSTRAINT daily_repo_issue_activities_pk PRIMARY KEY (id)
);

-- Table: daily_repo_issues
CREATE TABLE daily_repo_issues (
    id int NOT NULL AUTO_INCREMENT,
    repo_id int NOT NULL,
    date date NOT NULL,
    opened_count int NULL DEFAULT 0 COMMENT 'Number of issues opened that day',
    closed_count int NULL DEFAULT 0 COMMENT 'Number of issues closed that day',
    comment_count int NULL DEFAULT 0 COMMENT 'Number of comments made on issues that day',
    avg_resolution_sec int NULL DEFAULT 0,
    UNIQUE INDEX daily_repo_issues_ak_1 (repo_id,date),
    CONSTRAINT daily_repo_issues_pk PRIMARY KEY (id)
);

-- Table: daily_repo_releases
CREATE TABLE daily_repo_releases (
    id int NOT NULL AUTO_INCREMENT,
    repo_id int NOT NULL,
    date date NOT NULL,
    release_count int NOT NULL DEFAULT 0,
    UNIQUE INDEX daily_repo_releases_ak_1 (date,repo_id),
    CONSTRAINT daily_repo_releases_pk PRIMARY KEY (id)
);

-- Table: daily_repo_stars
CREATE TABLE daily_repo_stars (
    id int NOT NULL AUTO_INCREMENT,
    repo_id int NOT NULL,
    date date NOT NULL,
    star_count int NOT NULL DEFAULT 0,
    UNIQUE INDEX daily_repo_stars_ak_1 (repo_id,date),
    CONSTRAINT daily_repo_stars_pk PRIMARY KEY (id)
);

-- Table: daily_stackoverflow_questions
CREATE TABLE daily_stackoverflow_questions (
    id int NOT NULL AUTO_INCREMENT,
    repo_id int NOT NULL,
    date date NOT NULL,
    question_count int NOT NULL DEFAULT 0,
    answer_sum int NOT NULL DEFAULT 0,
    score_sum int NOT NULL DEFAULT 0,
    view_sum int NOT NULL DEFAULT 0,
    UNIQUE INDEX daily_stackoverflow_questions_ak_1 (repo_id,date),
    CONSTRAINT daily_stackoverflow_questions_pk PRIMARY KEY (id)
);

-- Table: forks
CREATE TABLE forks (
    id int NOT NULL AUTO_INCREMENT,
    date date NOT NULL,
    login varchar(255) NOT NULL,
    repo_id int NOT NULL,
    CONSTRAINT forks_pk PRIMARY KEY (id)
);

-- Table: issues
CREATE TABLE issues (
    id int NOT NULL AUTO_INCREMENT,
    opened_date date NOT NULL,
    resolved_date date NULL,
    reporter varchar(255) NULL,
    resolver varchar(255) NULL,
    resolution_duration_sec int NULL,
    repo_id int NOT NULL,
    CONSTRAINT issues_pk PRIMARY KEY (id)
);

-- Table: releases
CREATE TABLE releases (
    id int NOT NULL AUTO_INCREMENT,
    date date NOT NULL,
    login varchar(255) NOT NULL,
    repo_id int NOT NULL,
    CONSTRAINT releases_pk PRIMARY KEY (id)
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

-- Table: stars
CREATE TABLE stars (
    id int NOT NULL AUTO_INCREMENT,
    date date NOT NULL,
    login varchar(255) NOT NULL,
    repo_id int NOT NULL,
    CONSTRAINT stars_pk PRIMARY KEY (id)
);

-- foreign keys
-- Reference: Copy_of_daily_repo_issues_repos (table: daily_repo_commits)
ALTER TABLE daily_repo_commits ADD CONSTRAINT Copy_of_daily_repo_issues_repos FOREIGN KEY Copy_of_daily_repo_issues_repos (repo_id)
    REFERENCES repos (id);

-- Reference: Copy_of_daily_repo_stars_repos (table: daily_repo_contributions)
ALTER TABLE daily_repo_contributions ADD CONSTRAINT Copy_of_daily_repo_stars_repos FOREIGN KEY Copy_of_daily_repo_stars_repos (repo_id)
    REFERENCES repos (id);

-- Reference: commits_repos (table: commits)
ALTER TABLE commits ADD CONSTRAINT commits_repos FOREIGN KEY commits_repos (repo_id)
    REFERENCES repos (id);

-- Reference: daily_developer_stats_categories (table: daily_developer_stats)
ALTER TABLE daily_developer_stats ADD CONSTRAINT daily_developer_stats_categories FOREIGN KEY daily_developer_stats_categories (category_id)
    REFERENCES categories (id);

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

-- Reference: forks_repos (table: forks)
ALTER TABLE forks ADD CONSTRAINT forks_repos FOREIGN KEY forks_repos (repo_id)
    REFERENCES repos (id);

-- Reference: issues_repos (table: issues)
ALTER TABLE issues ADD CONSTRAINT issues_repos FOREIGN KEY issues_repos (repo_id)
    REFERENCES repos (id);

-- Reference: list_repos_lists (table: category_repos)
ALTER TABLE category_repos ADD CONSTRAINT list_repos_lists FOREIGN KEY list_repos_lists (category_id)
    REFERENCES categories (id);

-- Reference: list_repos_repos (table: category_repos)
ALTER TABLE category_repos ADD CONSTRAINT list_repos_repos FOREIGN KEY list_repos_repos (repo_id)
    REFERENCES repos (id);

-- Reference: releases_repos (table: releases)
ALTER TABLE releases ADD CONSTRAINT releases_repos FOREIGN KEY releases_repos (repo_id)
    REFERENCES repos (id);

-- Reference: stars_repos (table: stars)
ALTER TABLE stars ADD CONSTRAINT stars_repos FOREIGN KEY stars_repos (repo_id)
    REFERENCES repos (id);

-- End of file.

