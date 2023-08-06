# Hartufo
A Python toolkit for data-driven HRTF research

## Installation

The `hartufo` package is configured to be installable with `pip`, but because of its premature state, no releases are made and pushed to PyPI yet. Instead, you should install it from this Git repository.

### Using Prereleases
If you want to simply install a tagged prerelease (currently the latest is `v0.5.0`), you can 

- `pip install [--user] https://github.com/jpauwels/hartufo/archive/refs/tags/v0.5.0.zip`

### Development Setup
To contribute, it is best to check out the repo and install it in "editable mode" with the following procedure.

- `git clone https://github.com/jpauwels/hartufo/`
- `cd hartufo`
- `pip install [--user] -e .`

You can then simply update the package by calling `git pull` whenever the repo has been updated or even make your own edits and start contributing.
