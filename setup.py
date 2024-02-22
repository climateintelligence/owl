#!/usr/bin/env python

"""The setup script."""

import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, "README.rst")).read()
CHANGES = open(os.path.join(here, "CHANGES.rst")).read()
REQUIRES_PYTHON = ">=3.6.0"

about = {}
with open(os.path.join(here, "owl", "__version__.py"), "r") as f:
    exec(f.read(), about)

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

requirements = [line.strip() for line in open("requirements.txt")]

dev_reqs = [line.strip() for line in open("requirements_dev.txt")]

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX",
    "Programming Language :: Python",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
    "License :: OSI Approved :: Apache Software License",
]

setup(
    name="owl",
    version=about["__version__"],
    description="Heatwave magnitude index and warm nights",
    long_description=README + "\n\n" + CHANGES,
    long_description_content_type="text/x-rst",
    author=about["__author__"],
    author_email=about["__email__"],
    url="https://github.com/climateintelligence/owl",
    python_requires=REQUIRES_PYTHON,
    classifiers=classifiers,
    license="Apache Software License 2.0",
    keywords="wps pywps birdhouse owl",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    extras_require={
        "dev": dev_reqs,  # pip install ".[dev]"
    },
    entry_points={
        "console_scripts": [
            "owl=owl.cli:cli",
        ]
    },
)
