Major Changes of the Fork
=========================

The purpose of this page is to illustrate some of the main features that
``scikit-learn-tree`` provides compared to ``scikit-learn``. It assumes a
an understanding of core package ``scikit-learn`` and also decision trees
models. Please refer to our :ref:`installation instructions
<fork-installation-instructions>` for installing ``scikit-learn-tree``.

Scikit-learn-tree though operates as a stand-in for upstream ``scikit-learn``.
It is used in packages exactly the same way and will support all features
in the corresponding version of ``scikit-learn``. For example, if you
are interested in features of ``scikit-learn`` in v1.2.2 for ``NearestNeighbors`` algorithm,
then if ``scikit-learn-tree`` has a version release of v1.2.2, then it will have
all those features. 

The breaking API changes will be with respect to anything in the ``tree`` submodule,
and related Forest ensemble models. See below for a detailed list of breaking changes.

See: https://scikit-learn.org/ for documentation on scikit-learn main.

Our Philosophy
--------------
Our design philosophy with this fork of ``scikit-learn`` is to maintain as few changes
as possible, such that incorporating upstream changes into the fork requires minimal effort.

Candidate changes and PRs accepted into the fork are those that:

- improve compatability with upstream ``scikit-learn`` main
- enable improved extensibility of tree models

Decision tree generalizations
-----------------------------

``Scikit-learn`` provides an axis-aligned :class:`~sklearn.tree.DecisionTreeClassifier`
decision tree model (classifier and regressor), which has a few fundamental limitations
that prevent 3rd parties from utilizing the existing class, without forking a large
amount of copy/pasted Python and Cython code. We highlight those limitations here
and then describe how we generalize that limitation.

Cython Internal Private API:

Note, the Cython API for scikit-learn is still not a publicly supported API, so it may
change without warning.

- leaf and split nodes: These nodes are treated the same way and there is no internal
  API for setting them differently. Quantile trees and causal trees inherently generalize
  how leaf nodes are set.
- Criterion class: The criterion class currently assumes a supervised learning interface.
    - Our fix: We implement a ``BaseCriterion`` object that provides an abstract API for unsupervised
      criterion.
- Splitter class: The splitter clas currently assumes a supervised learning interface and
  does not provide a way of generalizing the way split candidates are proposed.
    - Our fix: We implement a ``BaseSplitter`` object that provides an abstract API for unsupervised
      splitters and also implement an API to allow generalizations of the ``SplitRecord`` struct and
      ``Splitter.node_split`` function. For example, this enables oblique splits to be considered.
- Tree class: The tree class currently assumes a supervised learning interface and does not
  provide a way of generalizing the type of tree.
    - Our fix: We implementa ``BaseTree`` object that provides an abstract API for general tree models
      and also implement an API that allows generalization of the type of tree. For example,
      oblique trees are trivially implementable as an extension now.
- stopping conditions for splitter: Currently, the ``Splitter.node_split`` function has various
  stopping conditions for the splitter based on hyperparameters. It is plausible that these conditions
  may be extended. For example, in causal trees, one may want the splitter to also account for
  a minimal degree of heterogeneity (i.e. variance) in its children nodes. 

Python API:

- ``sklearn.tree.BaseDecisionTree`` assumes the underlying tree model is supervised: The ``y``
  parameter is required to be passed in, which is not necessary for general tree-based models.
  For example, an unsupervised tree may pass in ``y=None``.
    - Our fix: We fix this API, so the ``BaseDecisionTree`` is subclassable by unsupervised
      tree models that do not require ``y`` to be defined.
- ``sklearn.tree.BaseDecisionTree`` does not provide a way to generalize the ``Criterion``, ``Splitter``
  and ``Tree`` Cython classes used: The current codebase requires users to define custom
  criterion and/or splitters outside the instantiation of the ``BaseDecisionTree``. This prevents
  users from generalizing the ``Criterion`` and ``Splitter`` and creating a neat Python API wrapper.
  Moreover, the ``Tree`` class is not customizable.
    - Our fix: We internally implement a private function to actually build the entire tree,
      ``BaseDecisionTree._build_tree``, which can be overridden in subclasses that customize the
      criterion, splitter, or tree, or any combination of them.

Overall, the existing tree models, such as :class:`~sklearn.tree.DecisionTreeClassifier`
and :class:`~sklearn.ensemble.RandomForestClassifier` all work exactly the same as they
would in ``scikit-learn`` main, but these extensions enable 3rd-party packages to extend
the Cython/Python API easily.

Next steps
----------

We have briefly covered how the tree submodule has changed with respect to ``scikit-learn``.
This enables packages to leverage these changes in developing more complex tree models
that may, or may not eventually be PRed into ``scikit-learn``. For example,

- `scikit-tree <https://docs.neurodata.io/scikit-tree/dev/index.html>`_ is a scikit-learn
  compatible package for more complex and advanced tree models.

If you are developing tree models, we encourage you to take a look at that package, or
if you have suggestions to make the tree submodule of our fork, ``scikit-learn-tree``
more 