import argparse
from pathlib import Path

from .detector import VenDetector
from .repo import Repository

def main() -> None:
    parser = argparse.ArgumentParser(prog="vendetect")

    parser.add_argument("TEST_REPO", type=str, help="path to the test repository")
    parser.add_argument("SOURCE_REPO", type=str, help="path to the source repository")

    args = parser.parse_args()

    with Repository.load(args.TEST_REPO) as test_repo, Repository.load(args.SOURCE_REPO) as source_repo:
        vend = VenDetector()
        for d in vend.detect(test_repo, source_repo):
            print(f"{d.test_repo!s}/{d.test_file!s} <-- {d.source_repo!s}/{d.source_file!s}")
