======
mrrvis
======


.. .. image:: https://img.shields.io/pypi/v/mrrvis.svg
..         :target: https://pypi.python.org/pypi/mrrvis

.. .. image:: https://img.shields.io/travis/AJPASHA/mrrvis.svg
..         :target: https://travis-ci.com/AJPASHA/mrrvis

.. .. image:: https://readthedocs.org/projects/mrrvis/badge/?version=latest
..         :target: https://mrrvis.readthedocs.io/en/latest/?version=latest
..         :alt: Documentation Status


.. .. image:: https://pyup.io/repos/github/AJPASHA/mrrvis/shield.svg
..      :target: https://pyup.io/repos/github/AJPASHA/mrrvis/
..      :alt: Updates

.. This is the image of the logo,  which is rendered in MRRVIS
.. image:: MRRVIS.png 

Description
===========
mrrvis is a visualisation tool for modular reconfigurable robotics (MRR) systems. It provides a simulation environment for MRR
systems in discrete lattices, including Square, Hexagonal, Triangular and Cubic lattices. Configurations are stored as graphs,
providing convenient means for verifying the connectivity of a shape and verifying the existence of an isomorphism between
two graphs, such as in checking that a designed configuration is the same as a predefined target.

Core Features (0.1.0)
=====================
    - Tools for graph based representations of robotic configurations in discrete lattices
    - An interface for modifying a configuration with respect to local collision and global configuration rules
    - Graphing tools to visualise configurations or, to visualise the configuration history of an environment
    - A means to compare configurations to identify if they are isomorphic to one another
    - Tools for defining new moves, and adding custom rules to the environment
    - Prebuilt Moves for Square and Hexagonal lattices

Dependancies
============
    - NumPy for matrix representations and scientific computing
    - Matplotlib for graphing configurations
    - mpl_toolkits.mplot3D for 3-Dimensional graphing

see requirements.txt for precise requirements

Installation
============
install with pip:
pip install git+https://github.com/AJPASHA/mrrvis@main

TODO: upload to pypi, 'pip install mrrvis'

Get Started
===========
To get started, import the following::

    from mrrvis.env import Environment

To create a square environment with the default moves and rules and make an action::
    
    state_0 = [[1,2],[2,2],[2,3],[3,3],[3,2],[3,1]]  # a simple configuration
    env = Environment(state_0, 'Square')   # create env
    env.render()    # visualise the environment
    env.step('slide', 0, 'N')    # perform a move
    env.render()    # visualise the new environment

For more in-depth demonstrations and tutorials, consult the notebooks in the demos folder of the repository

Testing
=======
For testing, use pytest::
    
    pytest tests

From the root directory of this repository. These tests cover all core functionality of the library. 

* Free software: MIT license
* Documentation: https://mrrvis.readthedocs.io.



Credits
-------
This package was created by Alexander Pasha at the University of Liverpool, 
under the supervision of Othon Michail


This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
