#! /usr/bin/env python
#
# Copyright (C) 2012-2020 Michael Waskom

DESCRIPTION = "scmfpt: Simple convex matrix factorization in pytorch"
#LONG_DESCRIPTION = """\
#"""

DISTNAME = 'scmfpt'
MAINTAINER = 'Giovanni Birolo'
#MAINTAINER_EMAIL = 'mwaskom@gmail.com'
#URL = 'https://seaborn.pydata.org'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/gbirolo/scmfpt/'
VERSION = '0.1'
PYTHON_REQUIRES = ">=3.6"

INSTALL_REQUIRES = [
    #'numpy>=1.16',
    #'pandas>=0.24',
    #'matplotlib>=3.0',
    'torch',
]

EXTRAS_REQUIRE = {
#    'all': [
#        'scipy>=1.2',
#        'statsmodels>=0.9',
#    ]
}


PACKAGES = [
    'scmfpt',
]


if __name__ == "__main__":
    from setuptools import setup

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        #author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        #maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        #long_description=LONG_DESCRIPTION,
        license=LICENSE,
        #url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        packages=PACKAGES,
        #classifiers=CLASSIFIERS
    )
