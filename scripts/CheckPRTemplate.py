# Copyright (c) 2025 PaddlePaddle Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys

import httpx

# ==============================
# PR Template Definition
# ==============================
REPO_TEMPLATE = {
    "FastDeploy": {
        "sections": [
            "## Motivation",
            "## Modifications",
            "## Usage or Command",
            "## Accuracy Tests",
            "## Checklist",
        ]
    }
}


# ==============================
# Utility Functions
# ==============================
def remove_comments(body):
    """Remove HTML-style comments (<!-- -->) from Markdown."""
    if not body:
        return ""
    comment_pattern = re.compile(r"<!--.*?-->", re.DOTALL)
    return comment_pattern.sub("", body).strip()


def check_section_content(body, section_titles):
    """Extract content between section headers."""
    results = {}
    valid_titles = [t for t in section_titles if t]

    for i, title in enumerate(valid_titles):
        next_title = valid_titles[i + 1] if i + 1 < len(valid_titles) else None

        if next_title:
            pattern = r"{}(.*?)(?={}|$)".format(re.escape(title), re.escape(next_title))
        else:
            pattern = r"{}(.*)".format(re.escape(title))  # Match until the end

        match = re.search(pattern, body, re.DOTALL | re.IGNORECASE)
        content = match.group(1).strip() if match else ""
        results[title] = content

    return results


def parse_checklist(section_content):
    """
    Parse a checklist section and return dict of items with checked status.
    Example return:
    {
        'Add at least a tag in the PR title.': False,
        'Format your code, run `pre-commit` before commit.': True,
        ...
    }
    """
    items = {}
    lines = section_content.splitlines()
    for line in lines:
        match = re.match(r"- \[( |x|X)\] (.+)", line)
        if match:
            checked = match.group(1).lower() == "x"
            item_text = match.group(2).strip()
            items[item_text] = checked
    return items


def check_pr_template(repo, body):
    """Check whether a PR description follows the expected template."""
    body = remove_comments(body)
    template_info = REPO_TEMPLATE.get(repo)

    if not template_info:
        print("[INFO] Repo '{}' not in REPO_TEMPLATE. Skipping check.".format(repo))
        return True, ""

    section_titles = template_info["sections"]
    results = check_section_content(body, section_titles)

    # Check missing sections
    missing = [sec for sec, content in results.items() if not content]
    messages = []

    if missing:
        if len(missing) == 1:
            messages.append("❌ Missing section: {}. Please complete it.".format(missing[0]))
        else:
            messages.append("❌ Missing sections: {}. Please complete them.".format(", ".join(missing)))

    # Check Checklist items if present
    checklist_content = results.get("## Checklist", "")
    if checklist_content:
        checklist_items = parse_checklist(checklist_content)
        unchecked = [item for item, checked in checklist_items.items() if not checked]
        if unchecked:
            messages.append("❌ The following checklist items are not completed:")
            for item in unchecked:
                messages.append(f"   - [ ] {item}")

    if messages:
        messages.append(
            "\n💡 **Tips for fixing:**\n"
            "1. Each PR must follow the standard FastDeploy PR template.\n"
            "2. Ensure every section (Motivation, Modifications, Usage, Accuracy Tests, Checklist) "
            "is clearly filled with relevant details.\n"
            "3. You can refer to the official PR example: "
            "https://github.com/PaddlePaddle/FastDeploy/blob/develop/.github/pull_request_template.md\n"
            "4. For missing parts, please describe briefly what was changed or verified.\n\n"
            "📩 If you have any questions, please contact **@yubaoku(EmmonsCurse)**"
        )
        return False, "\n".join(messages)

    return True, "✅ PR description template check passed."


def get_pull_request(org, repo, pull_id, token):
    """Fetch PR information from GitHub API."""
    url = f"https://api.github.com/repos/{org}/{repo}/pulls/{pull_id}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }

    response = httpx.get(url, headers=headers, follow_redirects=True, timeout=30)
    response.raise_for_status()
    return response.json()


# ==============================
# Main Entry
# ==============================
def main():
    org = os.getenv("AGILE_ORG", "PaddlePaddle")
    repo = os.getenv("AGILE_REPO", "FastDeploy")
    pull_id = os.getenv("AGILE_PULL_ID")
    token = os.getenv("GITHUB_API_TOKEN")

    if not pull_id or not token:
        print("❌ Environment variables AGILE_PULL_ID and GITHUB_API_TOKEN are required.")
        sys.exit(1)

    try:
        pr_info = get_pull_request(org, repo, pull_id, token)
    except Exception as e:
        print("❌ Failed to fetch PR info: {}".format(e))
        sys.exit(2)

    body = pr_info.get("body", "")
    title = pr_info.get("title", "")
    user = pr_info.get("user", {}).get("login", "unknown")

    print("🔍 Checking PR #{} by {}: {}".format(pull_id, user, title))

    ok, message = check_pr_template(repo, body)
    print(message)

    sys.exit(0 if ok else 7)


if __name__ == "__main__":
    main()
