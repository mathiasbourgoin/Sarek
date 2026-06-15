# Auto-Retarget PRs Workflow

This workflow automatically updates the base branch of dependent PRs when their target branch is merged.

## How It Works

When a PR is merged, this workflow:

1. **Identifies** all open PRs that were targeting the merged branch
2. **Updates** their base to point to the branch the PR was merged into
3. **Comments** on each updated PR to notify maintainers

## Example Scenario

```
main
 └── feature-A (PR #10) ✅ merged
      └── feature-B (PR #11) ⏸️ open, targeting feature-A
```

**Before PR #10 merge:**
- PR #11 targets `feature-A`

**After PR #10 is merged to `main`:**
- PR #11 automatically retargeted to `main`
- Comment added to PR #11 explaining the change

## Benefits

✅ **Merge Train Support** - Enables stacked PRs without manual retargeting  
✅ **Automatic** - No manual intervention needed  
✅ **Transparent** - Adds comments explaining what happened  
✅ **Safe** - Only runs on merged PRs, not closed ones

## Use Cases

- **Feature branches with sub-features** - Build features incrementally
- **Breaking changes** - Isolate risky changes in a feature branch
- **Long-running development** - Keep features organized in branches
- **Dependency chains** - PR B depends on PR A being merged first

## Permissions

This workflow requires:
- `pull-requests: write` - To update PR base branches and add comments
- `contents: read` - To access repository information

## Limitations

- Only processes open PRs (closed PRs are ignored)
- Fetches up to 100 PRs (should be sufficient for most repositories)
- Doesn't resolve merge conflicts (maintainers need to handle those)
