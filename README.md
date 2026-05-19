# multi-agent-rag

## Deployment readiness checklist

Run these checks before deployment to confirm version control state and latest files:

```bash
git fetch --all --prune
git status --short
git log --oneline -5
git ls-files
```

Ready for deployment when:

- `git status --short` has no output (clean working tree)
- Recent commit history looks correct
- Required files are present in `git ls-files`
