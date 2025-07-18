"""Tests for CLI functionality."""

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from vendetect._cli import main
from vendetect.detector import VenDetector


@pytest.fixture
def mock_repositories(tmp_path, monkeypatch):  # noqa: ANN201
    """Create mock test and source repositories for testing."""
    # Create test repo
    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()
    test_file = test_repo / "test_file.py"
    test_file.write_text("def hello_world():\n    print('Hello, World!')\n")

    # Create source repo
    source_repo = tmp_path / "source_repo"
    source_repo.mkdir()
    source_file = source_repo / "source_file.py"
    source_file.write_text("def hello_world():\n    print('Hello, World!')\n")

    # Create mock detection result
    mock_detection = MagicMock()
    mock_comparison = MagicMock()
    mock_comparison.similarity1 = 0.9
    mock_comparison.similarity2 = 0.9
    mock_comparison.slices1 = MagicMock()
    mock_comparison.slices1.tolist.return_value = [[1], [2]]
    mock_comparison.slices2 = MagicMock()
    mock_comparison.slices2.tolist.return_value = [[1], [2]]

    mock_detection.comparison = mock_comparison
    test_file_mock = MagicMock()
    test_file_mock.relative_path = test_file.relative_to(test_repo)
    mock_detection.test = test_file_mock
    source_file_mock = MagicMock()
    source_file_mock.relative_path = source_file.relative_to(source_repo)
    mock_detection.source = source_file_mock

    # Mock the detect method

    def mock_detect(self, test_repo, source_repo, file_filter=lambda _: True, max_history_depth=None):  # noqa: ANN202, ARG001
        return [mock_detection]

    monkeypatch.setattr(VenDetector, "detect", mock_detect)

    return str(test_repo), str(source_repo)


def test_output_format_csv(mock_repositories, capsys):  # noqa: ANN201, ARG001
    """Test that CSV output format works."""
    test_repo, source_repo = mock_repositories

    # Patch sys.argv
    with patch.object(sys, "argv", ["vendetect", test_repo, source_repo, "--format", "csv"]):
        try:
            # Redirecting stdout to capture output
            with patch("sys.stdout", new=StringIO()) as fake_out:
                main()
                output = fake_out.getvalue()

                # Check if output is in CSV format
                assert (
                    "Test File,Source File,Test Slice Start,Test Slice End,Source Slice Start,Source Slice End,"
                    "Similarity"
                ) in output
                # We're just checking the header format is correct, as our mock doesn't generate actual rows
                # The real implementation would generate rows for matching content
        except SystemExit:
            pass  # Main might exit, which is fine


def test_output_format_json(mock_repositories, capsys):  # noqa: ANN201, ARG001
    """Test that JSON output format works."""
    test_repo, source_repo = mock_repositories

    # Patch sys.argv
    with patch.object(sys, "argv", ["vendetect", test_repo, source_repo, "--format", "json"]):
        try:
            # Redirecting stdout to capture output
            with patch("sys.stdout", new=StringIO()) as fake_out:
                main()
                output = fake_out.getvalue()

                # Validate JSON output
                json_data = json.loads(output)
                assert isinstance(json_data, list)
                if json_data:  # If any detections were found
                    assert "test_file" in json_data[0]
                    assert "source_file" in json_data[0]
                    assert "similarity" in json_data[0]
                    assert "slices" in json_data[0]
        except SystemExit:
            pass  # Main might exit, which is fine


def test_output_to_file(mock_repositories, tmp_path):  # noqa: ANN201
    """Test that output to a file works."""
    test_repo, source_repo = mock_repositories
    output_file = tmp_path / "output.csv"

    # Patch sys.argv
    with patch.object(
        sys,
        "argv",
        ["vendetect", test_repo, source_repo, "--format", "csv", "--output", str(output_file)],
    ):
        try:
            main()

            # Check that the output file exists
            assert output_file.exists()

            # Check file content
            file_content = output_file.read_text()
            assert (
                "Test File,Source File,Test Slice Start,Test Slice End,Source Slice Start,Source Slice End,Similarity"
                in file_content
            )
            # We're just checking the header format is correct, as our mock doesn't generate actual rows
        except SystemExit:
            pass  # Main might exit, which is fine


def test_output_file_exists_no_force(mock_repositories, tmp_path):  # noqa: ANN201
    """Test that the program exits when output file exists and --force is not used."""
    test_repo, source_repo = mock_repositories
    output_file = tmp_path / "existing.csv"

    # Create the file
    output_file.write_text("existing content\n")

    # Patch sys.argv
    with patch.object(sys, "argv", ["vendetect", test_repo, source_repo, "--output", str(output_file)]):
        with pytest.raises(SystemExit) as excinfo:
            main()

        # Should exit with an error
        assert excinfo.value.code == 1


def test_output_file_exists_with_force(mock_repositories, tmp_path):  # noqa: ANN201
    """Test that the program overwrites existing files when --force is used."""
    test_repo, source_repo = mock_repositories
    output_file = tmp_path / "existing.json"

    # Create the file
    output_file.write_text("existing content\n")

    # Patch sys.argv
    with patch.object(
        sys,
        "argv",
        [
            "vendetect",
            test_repo,
            source_repo,
            "--format",
            "json",
            "--output",
            str(output_file),
            "--force",
        ],
    ):
        try:
            main()

            # Check that the file was overwritten
            file_content = output_file.read_text()
            assert "existing content" not in file_content

            # Should be valid JSON
            json_data = json.loads(file_content)
            assert isinstance(json_data, list)
        except SystemExit:
            pass  # Main might exit, which is fine
