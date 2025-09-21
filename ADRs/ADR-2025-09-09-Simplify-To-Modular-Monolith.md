# ADR: Simplify Architecture to Modular Monolith

## Context
Current microservices architecture introduces complexity, latency, and overhead for a coding assistant. Components like MCP Server, Web Server, Agents Service can be consolidated.

## Decision
Refactor to a modular monolith: Combine services into a single Python process with clear modules. Retain Docker for deployment but reduce inter-service calls.

## Consequences
- Pros: Simpler maintenance, reduced latency, easier development.
- Cons: Potential scaling limits; monitor and split if needed.

## Status
Proposed

## Date
2025-09-09