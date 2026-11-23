# Versioning and releases

The canonical project version is `[project].version` in `pyproject.toml`. Published releases use matching Git tags and GitHub Releases named `vX.Y.Z`.

This project is pre-1.0:

- Increment **PATCH** for backward-compatible fixes (`0.1.0` → `0.1.1`).
- Increment **MINOR** for new capabilities and breaking CLI/config changes (`0.1.0` → `0.2.0`).
- Use `1.0.0` only after the supported preprocessing, configuration, and model interfaces are considered stable.

## Release checklist

1. Work from the `minimal` default branch; do not replace or force-push the preserved `main` branch.
2. Update `pyproject.toml` and add a dated entry to `CHANGELOG.md`.
3. Run the documented test suite and relevant lint/type checks; do not include cohort data, patient outputs, or local clinical mappings.
4. Commit and push `minimal`, then create an annotated `vX.Y.Z` tag at that commit.
5. Push the tag and publish a GitHub Release using the matching changelog entry.
6. Read back the remote branch, tag, and release to verify the published version.
