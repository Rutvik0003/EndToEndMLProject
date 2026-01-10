from setuptools import setup, find_packages
from typing import List

def get_library(filepath:str)->List:

    libraries = []
    HYPHEN_E_DOT = "-e ."

    with open(filepath) as file:
        libraries = file.readlines()
        libraries = [lib.replace("\n","") for lib in libraries]

    if HYPHEN_E_DOT in libraries:
        libraries.remove(HYPHEN_E_DOT)

    return libraries
        


setup(
    name="End-to-End ML Project",
    version="0.0.1",
    author="Rutvik",
    packages=find_packages(),
    install_requires = get_library("requirements.txt")   
)