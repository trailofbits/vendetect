# Vendetect

<!--- BADGES: START --->
[![CI](https://github.com/trailofbits/vendetect/actions/workflows/tests.yml/badge.svg)](https://github.com/trailofbits/vendetect/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/vendetect.svg)](https://pypi.org/project/vendetect)
[![Packaging status](https://repology.org/badge/tiny-repos/python:vendetect.svg)](https://repology.org/project/python:vendetect/versions)
<!--- BADGES: END --->

A command-line tool for automatically detecting vendored and copy/pasted code between repositories.

## Description 🧑‍🎓

Vendetect helps identify copied or vendored code between repositories, making it easier to detect when code has been copied with or without attribution. The tool uses similarity detection algorithms to compare code files and highlight matching sections.

Key features:
- Compare code between two repositories (local or remote)
- Analyze specific subdirectories within repositories
- Identify files with similar code and display them side-by-side
- Show similarity percentages for matched code
- Filter by file types and adjust similarity thresholds
- Support for different programming languages through Pygments lexers
- Similarity is _not_ solely based upon symbol names; vendetect also considers semantics

## Installation 🚀

### Using pip

```bash
pip install vendetect
```

### Using [uv](https://docs.astral.sh/uv/guides/tools/)

```bash
uv tool install vendetect
```

### From source

Clone the repository and install:

```bash
git clone https://github.com/trailofbits/vendetect.git
cd vendetect
uv tool install .
```

### Development installation

For development with all dependencies:

```bash
git clone https://github.com/trailofbits/vendetect.git
cd vendetect
uv sync --group dev
source .venv/bin/activate
```

## Usage 🏃

### Basic usage

```bash
vendetect TEST_REPO SOURCE_REPO
```

Where:
- `TEST_REPO`: Path or URL to the repository you want to check for copied code
- `SOURCE_REPO`: Path or URL to the repository that is the potential source of the code

### Examples

```bash
# Compare two local repositories
vendetect /path/to/my/project /path/to/another/project

# Compare a local project with a remote repository
vendetect /path/to/my/project https://github.com/example/repo.git

# Compare only specific subdirectories within repositories
vendetect /path/to/my/project https://github.com/example/repo.git \
  --test-subdir src/components \
  --source-subdir lib/ui

# Filter by file types and adjust similarity threshold
vendetect /path/to/my/project /path/to/another/project \
  --type py --type js \
  --min-similarity 0.8
```

### Options

```
--format FORMAT              Output format: rich, csv, or json (default=rich)
--output OUTPUT              Output file path (default: stdout)
--force                      Force overwrite of existing output file
--type FILE_TYPES, -t        File extension to consider (can be used multiple times)
--min-similarity THRESHOLD   Minimum similarity threshold (range: 0.0-1.0, default: 0.5)
--test-subdir DIR, -ts       Subdirectory within TEST_REPO to analyze
--source-subdir DIR, -ss     Subdirectory within SOURCE_REPO to analyze
--incremental                Enable incremental result reporting
--batch-size SIZE            Number of files to process per batch (default: 100)
--max-history-depth DEPTH    Maximum commit history depth (default: -1 = entire history)
--log-level LEVEL            Sets the log level (default=INFO)
--debug                      Equivalent to --log-level=DEBUG
--quiet                      Equivalent to --log-level=CRITICAL
```

### Advanced Features

#### Subdirectory Analysis
When working with large repositories, you can focus analysis on specific subdirectories:

```bash
# Analyze only the src/ directory in both repositories
vendetect /path/to/my/project /path/to/another/project \
  --test-subdir src --source-subdir src

# Compare frontend code in one repo with backend in another
vendetect /path/to/frontend-repo /path/to/backend-repo \
  --test-subdir client/src --source-subdir server/utils
```

This is particularly useful for:
- Focusing on relevant code sections
- Reducing analysis time for large repositories
- Comparing similar modules across different project structures

#### File Type Filtering
Control which files are analyzed by specifying file extensions:

```bash
# Only analyze Python files
vendetect /path/to/my/project /path/to/another/project --type py

# Analyze multiple file types
vendetect /path/to/my/project /path/to/another/project --type py --type js --type ts
```

#### Similarity Thresholds
Adjust the minimum similarity threshold to filter results:

```bash
# Show only high-confidence matches (80% similarity or higher)
vendetect /path/to/my/project /path/to/another/project --min-similarity 0.8

# Show all potential matches (lower threshold)
vendetect /path/to/my/project /path/to/another/project --min-similarity 0.3
```

### Output Formats

Vendetect supports three output formats:

1. **rich** (default): Interactive console output with syntax highlighting and side-by-side code comparison
2. **csv**: Comma-separated values format with columns for Test File, Source File, Test Slice Start, Test Slice End, Source Slice Start, Source Slice End, and Similarity
3. **json**: JSON format with detailed information about each detection, including file paths, similarity scores, and matched code slices

Example using CSV output:
```bash
vendetect /path/to/my/project /path/to/another/project --format csv --output results.csv
```

Example using JSON output:
```bash
vendetect /path/to/my/project /path/to/another/project --format json --output results.json
```

## How it works 🧐

Vendetect uses a combination of techniques to identify similar code:

1. It fingerprints all source code files in both repositories based upon their semantics rather than syntax
2. For each file pair, it computes a similarity score
3. It identifies specific sections (slices) of code that match between files
4. Results are presented in a rich output format with side-by-side comparison

The tool can handle:
- Local file system repositories
- Git repositories (with history support)
- Remote git repositories (automatically cloned for analysis)

## Requirements 🛒

- Python 3.11 or higher
- Git (optional, for repository history analysis)

## Contributing 🧑‍💻

Contributions are welcome! Check out the [issues](https://github.com/trailofbits/vendetect/issues) for ideas on where to start.

### Development setup

```bash
# Install development dependencies
make dev

# Source virtual env
make test

# Lint code & Type checking
make lint
```

## Contact 💬

If you'd like to file a bug report or feature request, please use our
[issues](https://github.com/trailofbits/deptective/issues) page.
Feel free to contact us or reach out in
[Empire Hacking](https://slack.empirehacking.nyc/) for help using or extending Vendetect.

## License 📝

This utility was developed by [Trail of Bits](https://www.trailofbits.com/).

This program is free software: you can redistribute it and/or modify
it under the terms of the [GNU Affero General Public License](LICENSE) as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

[Contact us](mailto:opensource@trailofbits.com) if you're looking for an
exception to the terms.

© 2025, Trail of Bits.
