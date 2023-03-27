**scikit-learn-tree**
=====================
``scikit-learn-tree`` is a maintained fork of scikit-learn, which advances the tree submodule, while staying in-line
with changes from upstream scikit-learn. It is an exact stand-in for ``sklearn`` in package imports, but is
released under the name ``scikit-learn-tree`` to avoid confusion.

It is currently maintained by a team of volunteers.

The upstream package **scikit-learn** is a Python module for machine learning built on top of
SciPy and is distributed under the 3-Clause BSD license. Refer to their website for all documentation
needs: https://scikit-learn.org.

Why a fork?
-----------
Currently, the scikit-learn tree submodule is difficult to extend. Requests to modularize
and improve the extensibility of the code is currently unsupported, or may take a long time.
The desire for advanced tree models that also leverage the robustness of scikit-learn is desirable.

However, "hard-forking" via copy/pasting the explicit Python/Cython code into another tree package
altogether is undesirable because it results in a tree codebase that is inherently different
and not compatible with ``scikit-learn``. For example, `quantile-forests <https://github.com/zillow/quantile-forest>`_,
and `EconML <https://github.com/py-why/EconML>`_ do this, and their current tree submodules
cannot take advantage of improvements made in upstream ``scikit-learn``.

Maintaining a "soft-fork" of ``scikit-learn`` in the form of a repository fork allows us to develop
a separate package that serves as a stand-in for ``sklearn`` in any package, extends the tree submodule
and can also be synced with upstream changes in ``scikit-learn``. This enables this fork to always
take advantage of improvements made in ``scikit-learn`` main upstream, while providing a customizable
tree API.

Contents
--------

.. toctree::
   :maxdepth: 1

   Changelog of Fork<fork_changelog>
   Installation<fork_install>

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
