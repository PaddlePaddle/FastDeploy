# GitHub Copilot Custom Review Instructions

When reviewing code, focus on:

## Security Critical Issues
- Check for hardcoded secrets, API keys, or credentials
- Look for SQL injection and XSS vulnerabilities
- Verify proper input validation and sanitization
- Review authentication and authorization logic

## Performance Red Flags
- Identify N+1 database query problems
- Spot inefficient loops and algorithmic issues
- Check for memory leaks and resource cleanup
- Review caching opportunities for expensive operations

## Code Quality Essentials
- Functions should be focused and appropriately sized
- Use clear, descriptive naming conventions
- Ensure proper error handling throughout

## Review Style
- Be specific and actionable in feedback
- Explain the "why" behind recommendations
- Acknowledge good patterns when you see them
- Ask clarifying questions when code intent is unclear

Always prioritize security vulnerabilities and performance issues that could impact users.

Always suggest changes to improve readability. For example, this suggestion seeks to make the code more readable and also makes the validation logic reusable and testable.

// Instead of:
if (user.email && user.email.includes('@') && user.email.length > 5) {
  submitButton.enabled = true;
} else {
  submitButton.enabled = false;
}

// Consider:
function isValidEmail(email) {
  return email && email.includes('@') && email.length > 5;
}

submitButton.enabled = isValidEmail(user.email);

## Description for pull request

- Please check the title of the Pull Request. It needs to follow the format of [CLASS]Title, for example, [BugFix] Fix memory leak of data processor. If the title is incorrect, provide suggestions on how the committer should modify it.
- Please check the description information of the Pull Request. At a minimum, it should explain why these modifications are being made in this Pull Request and what problem is being solved. If the committer hasn't written the corresponding information or the information is incomplete, prompt the committer to make modifications.

## Language
- For repository members, please reply directly in Simplified Chinese; for external Contributors, reply in English.
