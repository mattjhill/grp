from distutils.core import setup

import my_grp

setup(
    name="my_grp",
    version=my_grp.__version__,
    author="Matthew Hill",
    author_email="matthew.hill@yale.edu",
    url="https://github.com/mjhgit/grp",
    packages=["my_grp"],
    description="Pure Python Generalized Rybicki Press Algorithm",
)