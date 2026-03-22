# RunPod Git Setup

This note covers the git steps for cloning your fork on RunPod and checking out the `feature_recovery` branch.

## Clone the repo on RunPod

From your RunPod workspace:

```bash
cd /workspace
git clone https://github.com/Adefioye/spd.git koko_spd
cd /workspace/koko_spd
```

## Check out `feature_recovery`

If the branch already exists on GitHub:

```bash
git fetch origin
git checkout feature_recovery
```

If `git checkout feature_recovery` does not find a local branch yet, create the local tracking branch explicitly:

```bash
git fetch origin
git checkout -b feature_recovery origin/feature_recovery
```

You can verify the active branch with:

```bash
git branch --show-current
```

Expected output:

```bash
feature_recovery
```

## One-command version

If you want to clone and switch branches in one flow:

```bash
cd /workspace
git clone https://github.com/Adefioye/spd.git koko_spd
cd /workspace/koko_spd
git fetch origin
git checkout -b feature_recovery origin/feature_recovery
```

## If the branch does not exist remotely

Check the remote branches:

```bash
git branch -r
```

If you do not see `origin/feature_recovery`, then the branch has not been pushed yet. In that case, push it from your local machine first, then rerun:

```bash
git fetch origin
git checkout -b feature_recovery origin/feature_recovery
```

## After checkout

Set up the environment from the repo root:

```bash
bash setup_env.sh
source .venv/bin/activate
```
