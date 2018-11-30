# ----------------------------------------------------------
# Author: Wheeler Earnest
#
# Project: Complexnet
# Setup file
# ------------------------------------------------------------

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="complexnet",
    version="0.1.6",
    author="Wheeler Earnest",
    author_email="jwheelerearnest@gmail.com",
    description="complex keras layers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WheelerEarnest/Complexnet",
    packages = setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
)
