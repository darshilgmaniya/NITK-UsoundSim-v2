# Contributing

## Where things go
| Change | Folder | Also update |
|---|---|---|
| Your team's physics or code | `teams/Tn_<name>/` | its tests, `CHANGELOG.md` |
| Joining modules | `integration/NN_<a>_<b>_relay/` | data dictionary, tests, CI |
| Explanations and reports | `docs/` | `docs/README.md` index |

## Workflow
1. Branch: `feature/<short-name>` or `fix/<short-name>`.
2. Make the change with a test that fails before and passes after.
3. Run the three suites (commands in `docs/11_DEVELOPER_GUIDE.md`, section 3).
4. If physics or defaults changed: re-run `verification/run_verification.py`, update `docs/08_VERIFICATION_REPORT.md` and `docs/09_PERFORMANCE.md`.
5. Add an entry to `CHANGELOG.md`.
6. Open a pull request with the test output; one reviewer from the affected team.

## Commit messages
`module: short imperative summary`, for example `acoustic_propagation: add element directivity option`.

## Rules that protect the other teams
- Do not change shapes, units, key names or row order of a hand-over file without a note in the changelog and a notice to the receiving team.
- Do not import one module from another; use the integration layer.
- Keep module READMEs accurate for what the module does on its own.

## Reporting a problem
Include: command run, full error text, `python --version`, operating system, and the output of `pip list | grep -i -E "numpy|scipy|matplotlib"`.
