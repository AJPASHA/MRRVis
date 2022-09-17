#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
import mrrvis

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy>=1.21', 'matplotlib>=3.5'] # Requirements might be a little conservative

test_requirements = ['pytest>=3', ]

setup(
    author="Alexander Pasha",
    author_email='alexanderjpasha@outlook.com',
    python_requires='>=3.9',
    # package_dir= {"":"mrrvis"},
    packages=find_packages(include=['mrrvis*']),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
    ],
    description="A visualisation tool for modular reconfigurable robotics systems",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='mrrvis',
    name='mrrvis',
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/AJPASHA/mrrvis',
    version='0.1.0',
    zip_safe=False,
)
