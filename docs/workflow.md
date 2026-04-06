# Team Workflow

## Branching Strategy

- `main` — stable, protected. No direct pushes.
- `feat/<description>` — new features
- `fix/<description>` — bug fixes
- `docs/<description>` — documentation only
- `refactor/<description>` — code restructure, no behavior change
- `chore/<description>` — tooling, deps, config

## Pull Request Rules

- PRs required for all changes to `main`
- Minimum **1 approval** required
- Keep PRs focused — one concern per PR

## Adding a Homework Package

1. Create a new package under `src/`:

```bash
cd src
ros2 pkg create --build-type ament_python homework_02_<topic> --dependencies rclpy
```

2. Add your node under `src/homework_02_<topic>/homework_02_<topic>/`
3. Add a launch file if needed under `src/puzzlebot_bringup/launch/`
4. Update `src/puzzlebot_bringup/package.xml` to add the new exec_depend
5. Open a PR from `feat/homework-02-<topic>`

## Adding Shared Code

- Place reusable helpers in `src/shared_utils/shared_utils/`
- Add the dependency to any package that uses it
- Keep shared_utils free of homework-specific logic

## Commit Message Convention

```
<type>(<scope>): <short summary>

Types: feat, fix, docs, refactor, chore, test
Scope: package name or area (e.g. hw01, bringup, description)

Examples:
feat(hw01): add circular trajectory TF publisher
fix(description): correct wheel mesh path in URDF
docs(setup): add WSL2 display instructions
```
