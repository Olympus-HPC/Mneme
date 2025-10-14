#!/usr/bin/env bash

set -e

echo "CI_COMMIT_REF_NAME ${CI_COMMIT_REF_NAME}"
# Fetch the PR ID from the branch name.
PR_INFO=$(curl -s -L -H "Authorization: Bearer $GITHUB_TOKEN" \
               -H "Accept: application/vnd.github+json" \
               -H "X-GitHub-Api-Version: 2022-11-28" \
               "https://api.github.com/repos/Olympus-HPC/Mneme/pulls?head=Olympus-HPC:${CI_COMMIT_REF_NAME}")

echo "PR_INFO = ${PR_INFO}"

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

RESULTS_COMMIT="Artifacts PR ${PR_ID} commit ${CI_COMMIT_SHORT_SHA}"

# Run the selected benchmark
SECONDS=0
bash -c "${BENCHMARKS}"
END_TIME=$SECONDS
COMMENT="${RESULTS_COMMIT}: ${COMMENTS_BODY} ran in ${END_TIME} seconds\n<p>"

# Post the comment to the GitHub PR.
curl -L -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/Olympus-HPC/Mneme/issues/${PR_ID}/comments" \
  -d "{\"body\": \"${COMMENT}\"}"
