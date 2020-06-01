# Copyright 2020 Zhi Huang.  All rights reserved
# Created on Mon Feb 10 17:57:08 2020
# Author: Zhi Huang, Purdue University
#
# The original code came with the following disclaimer:
#
# This software is provided "as-is".  There are no expressed or implied
# warranties of any kind, including, but not limited to, the warranties
# of merchantability and fitness for a given application.  In no event
# shall Zhi Huang be liable for any direct, indirect, incidental,
# special, exemplary or consequential damages (including, but not limited
# to, loss of use, data or profits, or business interruption) however
# caused and on any theory of liability, whether in contract, strict
# liability or tort (including negligence or otherwise) arising in any way
# out of the use of this software, even if advised of the possibility of
# such damage.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="biolearns",
    version="0.0.45",
    author="Zhi Huang",
    author_email="huang898@purdue.edu",
    description="BioLearns: Computational Biology and Bioinformatics Toolbox in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://biolearns.com",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
