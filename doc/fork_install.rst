.. _fork-installation-instructions:

============================
Installing scikit-learn-tree
============================

Scikit-learn-tree is a maintained fork of scikit-learn, which extends the
tree submodule in a few ways documented in :ref:`changelog of the fork
<fork_changelog>`. 

We release versions of scikit-learn-tree in an analagous fashion to
scikit-learn main. Due to maintenance resources, we only release on PyPi
and recommend therefore installing with ``pip``.

There are different ways to install scikit-learn-tree:

  * :ref:`Install the latest official release <install_fork_release>`. This
    is the best approach for most users. It will provide a stable version
    and pre-built packages are available for most platforms.
    
  * :ref:`Building the package from source
    <install_source>`. This is best for users who want the
    latest-and-greatest features and aren't afraid of running
    brand-new code. This is also needed for users who wish to contribute to the
    project.

.. _install_fork_release:

Installing the latest release
=============================
We release wheels for common distributions and this is thus installable via pip.

.. prompt:: bash $
  
  pip install scikit-learn-tree

This will install ``scikit-learn-tree`` under the namespace of ``sklearn``, which then
can be used as a stand-in for any package that relies on the public API of ``sklearn``.

For example, any usage of ``scikit-learn`` is preserved with ``scikit-learn-tree``

  >>> # the sklearn installed is that of scikit-learn-tree and is equivalent to scikit-learn
  >>> from sklearn.ensemble import RandomForestClassifier
  >>> clf = RandomForestClassifier(random_state=0)
  >>> X = [[ 1,  2,  3],  # 2 samples, 3 features
  ...      [11, 12, 13]]
  >>> y = [0, 1]  # classes of each sample
  >>> clf.fit(X, y)
  RandomForestClassifier(random_state=0)

.. _install_source:

Building from source
--------------------
If you are a developer and are interested in helping maintain, or add some new
features to the fork, the building from source instructions are exactly the same
as that of scikit-learn main, so please refer to `scikit-learn documentation <https://scikit-learn.org/stable/developers/advanced_installation.html#install-bleeding-edge>`_
for instructions on building from source.
