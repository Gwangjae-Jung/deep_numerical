---
trigger: always_on
---

# Git Workflow Rules

When performing Git operations in this repository, the following rules must be followed at all times.

## 1. Branch Structure

- `master`: Production branch. Do not commit directly to it or merge feature branches into it.
- `develop`: Development integration branch. Create it from `master`. If it does not exist, create it as follows:

```bash
git checkout master
git checkout -b develop
```

- Feature branches: All work must be done on a new branch created from `develop`.

## 2. Feature Branch Naming

- Format: `type/content`
- `content` must be written in lowercase English, use hyphens (`-`) between words, and remain short and descriptive.
- Examples:
  - `feat/login-form`
  - `fix/header-layout`
  - `refactor/user-service`

## 3. Type List (Used for Both Branches and Commits)

- `feat`: Add a new feature
- `fix`: Fix a bug
- `refactor`: Improve code structure without changing behavior
- `docs`: Update documentation
- `style`: Changes that do not affect code behavior, such as formatting or semicolons
- `test`: Add or modify tests
- `chore`: Other tasks such as build configuration or package management

## 4. Commit Messages

- Format: `type: content`
- Include a single space after the colon.
- `content` should be a concise one-line description of what was done and must not end with a period.
- Each commit should contain only one purpose. Changes with different purposes must be split into separate commits.
- Examples:
  - `feat: add login form`
  - `fix: fix header alignment issue`

## 5. Workflow

1. Create a feature branch from `develop`.

```bash
git checkout develop
git checkout -b feat/login-form
```

2. Commit changes following the commit message convention.

```bash
git commit -m "feat: add login form"
```

3. After completing the feature, merge it into `develop` using `--no-ff`.

```bash
git checkout develop
git merge --no-ff feat/login-form
```

4. The feature branch may be deleted after the merge is completed.

```bash
git branch -d feat/login-form
```

## 6. Prohibited Actions

- Do not commit feature work directly to `master` or `develop`.
- Do not omit `--no-ff` when merging feature branches (fast-forward merges are prohibited).
- Do not create feature branches from `master`. Always create them from `develop`.
- Do not use branch names or commit messages that violate these conventions.

## 7. When in Doubt

- If you are unsure which type to use, or if you need to perform a potentially irreversible operation such as merging into `master`, force-pushing, or deleting branches, ask the user for confirmation before proceeding.