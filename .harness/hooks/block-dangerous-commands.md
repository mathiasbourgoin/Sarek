---
name: block-dangerous-commands
description: Block destructive shell commands — rm -rf root/home, force push to main, SQL drops, chmod 777, pipe to shell.
event: PreToolUse
matcher: Bash
version: 1.0.0
timeout: 5000
---

# Block Dangerous Shell Commands

Intercepts Bash tool calls and rejects commands matching known destructive patterns. A non-zero exit blocks the tool call and shows the reason to the model.

## Blocked Patterns

| Pattern | Reason |
|---------|--------|
| `rm -rf /`, `rm -rf ~`, `rm -rf .` | Catastrophic recursive deletion of root, home, or working directory |
| `git push --force` / `git push -f` to main/master | Rewrites shared history on protected branches |
| `git reset --hard` | Discards uncommitted work irreversibly |
| `DROP TABLE`, `DROP DATABASE`, `TRUNCATE` | Destructive SQL — data loss |
| `chmod 777` | Removes all permission restrictions — security violation |
| `curl ... \| sh`, `wget ... \| sh` | Arbitrary remote code execution |

## Command

```command
#!/bin/bash
INPUT=$(cat -)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# rm -rf targeting root, home, or cwd
if echo "$COMMAND" | grep -qE 'rm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+(/|~|\.)(\s|$|;|\|)'; then
  echo "BLOCKED: 'rm -rf' targeting /, ~, or . is not allowed. Use a specific path."
  exit 1
fi
if echo "$COMMAND" | grep -qE 'rm\s+-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+(/|~|\.)(\s|$|;|\|)'; then
  echo "BLOCKED: 'rm -rf' targeting /, ~, or . is not allowed. Use a specific path."
  exit 1
fi

# git push --force / -f to main or master
if echo "$COMMAND" | grep -qE 'git\s+push\s+.*(-f|--force)' && echo "$COMMAND" | grep -qE '\b(main|master)\b'; then
  echo "BLOCKED: Force push to main/master is not allowed. Push to a feature branch instead."
  exit 1
fi

# git reset --hard (any target)
if echo "$COMMAND" | grep -qE 'git\s+reset\s+--hard'; then
  echo "BLOCKED: 'git reset --hard' discards work irreversibly. Ask the user for confirmation first."
  exit 1
fi

# Destructive SQL
if echo "$COMMAND" | grep -qiE '(DROP\s+TABLE|DROP\s+DATABASE|TRUNCATE)\b'; then
  echo "BLOCKED: Destructive SQL detected. Ask the user for confirmation first."
  exit 1
fi

# chmod 777
if echo "$COMMAND" | grep -qE 'chmod\s+777'; then
  echo "BLOCKED: 'chmod 777' removes all permission restrictions. Use a more restrictive mode."
  exit 1
fi

# Piping remote content to shell
if echo "$COMMAND" | grep -qE '(curl|wget)\s+.*\|\s*(ba)?sh'; then
  echo "BLOCKED: Piping remote content to shell is arbitrary code execution. Download and inspect first."
  exit 1
fi

exit 0
```

## Installed As

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "#!/bin/bash\nINPUT=$(cat -)\nCOMMAND=$(echo \"$INPUT\" | jq -r '.tool_input.command // empty')\nif echo \"$COMMAND\" | grep -qE 'rm\\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\\s+(/|~|\\.)([\\s;|]|$)'; then\n  echo \"BLOCKED: rm -rf targeting /, ~, or . is not allowed.\"\n  exit 1\nfi\nif echo \"$COMMAND\" | grep -qE 'rm\\s+-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\\s+(/|~|\\.)([\\s;|]|$)'; then\n  echo \"BLOCKED: rm -rf targeting /, ~, or . is not allowed.\"\n  exit 1\nfi\nif echo \"$COMMAND\" | grep -qE 'git\\s+push\\s+.*(-f|--force)' && echo \"$COMMAND\" | grep -qE '\\b(main|master)\\b'; then\n  echo \"BLOCKED: Force push to main/master is not allowed.\"\n  exit 1\nfi\nif echo \"$COMMAND\" | grep -qE 'git\\s+reset\\s+--hard'; then\n  echo \"BLOCKED: git reset --hard requires explicit user confirmation.\"\n  exit 1\nfi\nif echo \"$COMMAND\" | grep -qiE '(DROP\\s+TABLE|DROP\\s+DATABASE|TRUNCATE)\\b'; then\n  echo \"BLOCKED: Destructive SQL detected.\"\n  exit 1\nfi\nif echo \"$COMMAND\" | grep -qE 'chmod\\s+777'; then\n  echo \"BLOCKED: chmod 777 is not allowed.\"\n  exit 1\nfi\nif echo \"$COMMAND\" | grep -qE '(curl|wget)\\s+.*\\|\\s*(ba)?sh'; then\n  echo \"BLOCKED: Piping remote content to shell is not allowed.\"\n  exit 1\nfi\nexit 0",
            "description": "Block dangerous shell commands"
          }
        ]
      }
    ]
  }
}
```
