.. -*- mode: rst -*-

|Azure|_ |CirrusCI|_ |Codecov|_ |CircleCI|_ |Nightly wheels|_ |Black|_ |PythonVersion|_ |PyPi|_ |DOI|_ |Benchmark|_

.. |Azure| image:: https://dev.azure.com/scikit-learn/scikit-learn/_apis/build/status/scikit-learn.scikit-learn?branchName=main
.. _Azure: https://dev.azure.com/scikit-learn/scikit-learn/_build/latest?definitionId=1&branchName=main

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn/scikit-learn/tree/main.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn/scikit-learn

.. |CirrusCI| image:: https://img.shields.io/cirrus/github/scikit-learn/scikit-learn/main?label=Cirrus%20CI
.. _CirrusCI: https://cirrus-ci.com/github/scikit-learn/scikit-learn/main

.. |Codecov| image:: https://codecov.io/gh/scikit-learn/scikit-learn/branch/main/graph/badge.svg?token=Pk8G9gg3y9
.. _Codecov: https://codecov.io/gh/scikit-learn/scikit-learn

.. |Nightly wheels| image:: https://github.com/scikit-learn/scikit-learn/workflows/Wheel%20builder/badge.svg?event=schedule
.. _`Nightly wheels`: https://github.com/scikit-learn/scikit-learn/actions?query=workflow%3A%22Wheel+builder%22+event%3Aschedule

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue
.. _PythonVersion: https://pypi.org/project/scikit-learn/

.. |PyPi| image:: https://img.shields.io/pypi/v/scikit-learn
.. _PyPi: https://pypi.org/project/scikit-learn

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: https://github.com/psf/black

.. |DOI| image:: https://zenodo.org/badge/21369/scikit-learn/scikit-learn.svg
.. _DOI: https://zenodo.org/badge/latestdoi/21369/scikit-learn/scikit-learn

.. |Benchmark| image:: https://img.shields.io/badge/Benchmarked%20by-asv-blue
.. _`Benchmark`: https://scikit-learn.org/scikit-learn-benchmarks/

.. |PythonMinVersion| replace:: 3.8
.. |NumPyMinVersion| replace:: 1.17.3
.. |SciPyMinVersion| replace:: 1.5.0
.. |JoblibMinVersion| replace:: 1.1.1
.. |ThreadpoolctlMinVersion| replace:: 2.0.0
.. |MatplotlibMinVersion| replace:: 3.1.3
.. |Scikit-ImageMinVersion| replace:: 0.16.2
.. |PandasMinVersion| replace:: 1.0.5
.. |SeabornMinVersion| replace:: 0.9.0
.. |PytestMinVersion| replace:: 5.3.1
.. |PlotlyMinVersion| replace:: 5.10.0

.. image:: https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/logos/scikit-learn-logo.png
  :target: https://scikit-learn.org/

This is a maintained fork of scikit-learn, which advances the tree submodule, while staying in-line
with changes from upstream scikit-learn. It is currently maintained by a team of volunteers.

**scikit-learn** is a Python module for machine learning built on top of
SciPy and is distributed under the 3-Clause BSD license.

Website: https://scikit-learn.org

Installation
------------

Dependencies
~~~~~~~~~~~~

scikit-learn requires:

- Python (>= |PythonMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- SciPy (>= |SciPyMinVersion|)
- joblib (>= |JoblibMinVersion|)
- threadpoolctl (>= |ThreadpoolctlMinVersion|)

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of numpy and scipy,
the easiest way to install scikit-learn is using ``pip``::

    pip install -U scikit-learn

or ``conda``::

    conda install -c conda-forge scikit-learn

The documentation includes more detailed `installation instructions <https://scikit-learn.org/stable/install.html>`_.


Changelog
---------

See the `changelog <https://scikit-learn.org/dev/whats_new.html>`__
for a history of notable changes to scikit-learn.

Development
-----------

We welcome new contributors of all experience levels, specifically to maintain the fork.
Any contributions that make sure our fork is "better in-line" with scikit-learn upstream,
or improves the tree submodule in anyway will be appreciated.

The scikit-learn community goals are to be helpful, welcoming, and effective. The
`Development Guide <https://scikit-learn.org/stable/developers/index.html>`_
has detailed information about contributing code, documentation, tests, and
more. We've included some basic information in this README.

Important links
~~~~~~~~~~~~~~~

- Official source code repo: https://github.com/scikit-learn/scikit-learn
- Download releases: https://pypi.org/project/scikit-learn/
- Issue tracker: https://github.com/scikit-learn/scikit-learn/issues
