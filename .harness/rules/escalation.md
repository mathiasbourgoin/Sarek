---
name: escalation
description: Default escalation triggers — pause and ask the human before destructive or high-impact actions.
scope: global
category: safety
version: 1.0.0
---

# Default Escalation Triggers

Pause and ask the human for explicit confirmation before performing any of the following:

- **Destructive file operations:** `rm -rf`, mass file deletion, overwriting files outside the current task scope.
- **Destructive git operations:** `git reset --hard`, `git push --force` / `git push -f` to any branch, `git clean -f`.
- **Destructive SQL:** `DROP TABLE`, `DROP DATABASE`, `TRUNCATE`, `DELETE` without a `WHERE` clause.
- **Force-pushing** to any branch, including feature branches.
- **External API calls with side effects:** `POST`, `PUT`, `DELETE`, or `PATCH` to production endpoints.
- **CI/CD pipeline modifications:** Changing workflow files, build configs, deployment scripts, or pipeline triggers.
- **Auth and security changes:** Modifying permissions, access tokens, secrets, firewall rules, or auth configuration.
- **MCP server changes:** Installing, removing, or modifying MCP server configurations.
- **Shared infrastructure:** Any action affecting resources used by other people or services (databases, message queues, DNS, load balancers).
- **Cost threshold:** Any action exceeding a configurable cost threshold (default: warn on operations that may incur billing).
- **Properties file:** Any action listed in `kb/properties.md` as requiring human approval.

When escalating, state what you intend to do, why, and what the blast radius is. Do not proceed until the human confirms.
