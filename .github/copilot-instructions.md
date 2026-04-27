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
- For all Pull Requests, please confirm whether it is necessary to add, update, or delete documentation, and remind the committer to handle it accordingly.

## Logging Standards

### Log Channels
| Channel | Usage | Import |
|---------|-------|--------|
| Main Log | Core business logic, scheduling, request management | `from fastdeploy.logger import llm_logger` |
| Request Log | Request lifecycle and content tracking | `from fastdeploy.logger import log_request, RequestLogLevel` |
| Console Log | User-facing messages | `from fastdeploy.logger import console_logger` |

### Request Log Levels (FD_LOG_REQUESTS_LEVEL)
| Level | Enum | Description | Example Content |
|-------|------|-------------|-----------------|
| 0 | LIFECYCLE | Lifecycle start/end | Request creation, completion stats, streaming first/last send, abort |
| 1 | STAGES | Processing stages | Semaphore acquire/release, first token time, signal handling, cache task, preprocess time |
| 2 | CONTENT | Content and scheduling | Request params, scheduling info (enqueue/pull/finish), response content (truncated) |
| 3 | FULL | Complete raw data | Full request/response data, raw received request |

### Examples
```python
# Main log - core business logic
self.llm_logger.info("Engine started")
self.llm_logger.debug(f"abort targets finished: {req_ids}")

# Request log - request lifecycle
log_request(RequestLogLevel.LIFECYCLE, "request created: {req_id}", req_id=req_id)
log_request(RequestLogLevel.CONTENT, "response: {content}", content=response)
```

### Log Review Checklist
- **Channel selection**: Request-related logs should use `log_request`, system-level logs use `llm_logger`
- **Level selection**: Use `debug` for frequent operations, avoid flooding with `info` level logs
- **Request log level**: Refer to the Request Log Levels table above to select the appropriate level
- **New log files**: In principle, do not add new standalone log files, should be merged into existing log files

## Others
- 对于所有提交的PR，你提交的评论都使用中文语言，但需要注意，代码中的注释仍然需要使用英文
- 在你提交Pull Request时，需要注意代码风格要满足本Repo的设定，commit代码前需要`pip install pre-commit==4.2.0`并且执行`pre-commit install`
