"""Project setup file."""

from typing import List
from setuptools import find_packages, setup


HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    Returns a list of requirements from the given file.
    """
    requirements = []
    with open(file_path, "wb") as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="4.End-to-End Smart Sales Analysis & Forecasting System for E-Commerce",
    version="0.9.0",
    author="KrzysztofDK",
    author_email="krzysztof.d.kopytowski@gmail.com",
    packages=find_packages(),
    python_requires=">=3.10,<3.11",
    install_requires=get_requirements("requirements.txt"),
)
