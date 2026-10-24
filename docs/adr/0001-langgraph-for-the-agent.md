# ADR 0001: LangGraph for the copilot's orchestration

**Status:** accepted · **Date:** 2026-10-24

## Context
The copilot needs a loop: Claude picks tools, tools run, Claude answers, a check step verifies the numbers and can
send the answer back once. The steps must be visible, testable and traceable.

## Decision
Use **LangGraph** for the state machine, with **Claude called through the Anthropic SDK** inside the nodes.
Tools stay plain Python functions.

## Why
- The graph is explicit (nodes, edges, state), so it can be drawn, reviewed and traced step by step (MLflow autolog).
- Model-agnostic: the same graph works with the Anthropic API or Amazon Bedrock.
- Calling the SDK directly keeps the cost meter and Bedrock model mapping simple, and avoids extra wrapper layers.
- Widely used in industry, so new team members know it.

## Alternatives
- Plain tool loop: simplest, but no standard structure or tracing. Kept as a fallback runner with the same nodes.
- Strands Agents / Bedrock AgentCore: AWS-native, but ties the demo to AWS.
- Claude Agent SDK: strong for agents that work with files and code; more than a CRM chat needs.
- CrewAI: built for several cooperating agents; we have one.
