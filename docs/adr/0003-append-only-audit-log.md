# ADR 0003: Append-only, hash-chained audit log

**Status:** accepted · **Date:** 2026-10-24

## Decision
Every copilot step (question, each Claude call with tokens and cost, each tool call with its input, the check, the
answer) is written to an `events` table. Each row stores the SHA-256 of the previous row and its own content.

## Why
Sales recommendations must be explainable later ("why did it tell me to call this customer?"). Traces in MLflow are
for engineers and can be deleted; the audit log is the record. The hash chain makes edits or deletions visible
(`python -m fleetpilot audit-verify`).

## Rules
- No names, no contact data, no API keys: only pseudonymised customer and rep IDs.
- Insert only. Retention and backup follow the company's records policy.
