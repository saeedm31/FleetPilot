# ADR 0004: The agent reads a pseudonymised, read-only copy of the CRM

**Status:** accepted · **Date:** 2026-10-24

## Decision
The pipeline writes a small SQLite database with only what the copilot needs, without names. The agent opens it
read-only, behind an authorizer that allows SELECT on known tables only, one statement, a row cap and a time limit.

## Why
Agent-written SQL is untrusted input. Defense in depth: read-only file mode, an allow-list authorizer, statement checks,
and no personal data in the database at all. In production the same idea maps to read-only database views and a
service account without write rights.
