# Security and data protection

## Secrets
- No secrets in code or images. `ANTHROPIC_API_KEY`, `PSEUDONYM_KEY` come from the environment (`.env` locally, a
  secret store such as AWS Secrets Manager in production). `.env` is git-ignored.
- In the public demo the visitor's API key lives only in their browser session; it is never stored or logged.
- On AWS the app uses an IAM role for Amazon Bedrock: no key at all.

## Personal data (GDPR)
- The copilot, Claude and all logs see only pseudonymised customer IDs; names stay in a separate file on the rep side.
- The copilot declines requests for personal data (phone numbers, addresses).
- Data minimisation: the agent's database holds only what the copilot needs.

## LLM-specific risks
| Risk | Control |
|---|---|
| Prompt injection via data | System prompt treats tool output as data; tools return structured JSON; no tool can write |
| Harmful SQL | Read-only connection, allow-list authorizer (SELECT on known tables), one statement, row and time limits |
| Made-up numbers | Check step: every number must appear in a tool result, else one rewrite, and the result is shown and logged |
| Runaway cost | Step limit per question, token limits per call, monthly/session budget that stops new questions |
| Untraceable answers | Append-only hash-chained audit log + MLflow traces |

## Containers
Non-root user (uid 1000), slim base image, health checks, no secrets baked into images.

## Reporting
Please open a private security advisory on GitHub.
