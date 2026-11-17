
# Question 1: Database Schema Graph

## Dataset

- Source: GH Archive
- Historical period (Taiwan time):
  2026/10/25 00:00 – 2026/11/16 00:00
- Historical period (UTC):
  2026/10/24 16:00 – 2026/11/15 16:00
- Total tables: 8
- Total observed cohort events: 14,113

## Schema Graph

```mermaid
flowchart TB
    U["Users<br/>PK: user_id"]
    R["Repositories<br/>PK: repo_id"]
    P["PullRequests<br/>PK: pr_id<br/>FK: repo_id, author_id, merged_by_id"]
    I["Issues<br/>PK: issue_id<br/>FK: repo_id, author_id, pull_request_id, assignee_id"]
    IC["IssueComments<br/>PK: comment_id<br/>FK: issue_id, repo_id, author_id"]
    PR["PRReviews<br/>PK: review_id<br/>FK: pr_id, repo_id, reviewer_id"]
    RC["PRReviewComments<br/>PK: comment_id<br/>FK: pr_id, repo_id, author_id, pull_request_review_id, in_reply_to_id"]
    PS["Pushes<br/>PK: push_id<br/>FK: user_id, repo_id"]

    P -->|author_id, merged_by_id| U
    P -->|repo_id| R

    I -->|author_id, assignee_id| U
    I -->|repo_id| R
    I -->|pull_request_id| P

    IC -->|author_id| U
    IC -->|repo_id| R
    IC -->|issue_id| I

    PR -->|reviewer_id| U
    PR -->|repo_id| R
    PR -->|pr_id| P

    RC -->|author_id| U
    RC -->|repo_id| R
    RC -->|pr_id| P
    RC -->|pull_request_review_id| PR
    RC -->|in_reply_to_id| RC

    PS -->|user_id| U
    PS -->|repo_id| R
```
