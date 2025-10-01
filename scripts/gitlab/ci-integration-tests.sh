#!/usr/bin/env bash

set -e

echo "CI_COMMIT_REF_NAME ${CI_COMMIT_REF_NAME}"
# Fetch the PR ID from the branch name.
PR_INFO=$(curl -s -L -H "Authorization: Bearer $GITHUB_TOKEN" \
               -H "Accept: application/vnd.github+json" \
               -H "X-GitHub-Api-Version: 2022-11-28" \
               "https://api.github.com/repos/Olympus-HPC/Mneme/pulls?head=Olympus-HPC:${CI_COMMIT_REF_NAME}")

# Check if PR exists.
if [ -z "${PR_INFO}" ] || [ "$(echo "$PR_INFO" | jq length)" = "0" ]; then
  echo "No PR found for ref ${CI_COMMIT_REF_NAME}, exit"
  exit 0
fi

# Extract PR number.
PR_ID=$(echo "${PR_INFO}" | jq -r '.[0].number')
echo "Processing PR ${PR_ID}"

COMMENTS_INFO=$(curl -L \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/Olympus-HPC/Mneme/issues/${PR_ID}/comments")
COMMENTS_BODY=$(echo ${COMMENTS_INFO} | jq -r '.[].body')
if [[ "${COMMENTS_BODY}" == *"/run-laghos"* ]]; then
  echo "=> Run Laghos integration tests";
  BENCHMARKS="scripts/gitlab/ci-integration-laghos.sh"
else
  echo "=> Invalid command. Available commands: /run-{laghos}"
  exit 0
fi

# Run the selected benchmarks
(bash -c "${BENCHMARKS}")