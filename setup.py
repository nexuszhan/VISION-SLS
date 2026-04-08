from setuptools import setup, find_packages
from pathlib import Path
import os

repo_root = Path(__file__).resolve().parent

setup(
    name='safe-perception-sls',
    version='1.0',
    packages=find_packages(repo_root),
    install_requires=[],
)
