.. _contributing:

Contributing
============
CLA guidelines
-----------
Please be aware that we can only consider pull requests of authors who consider 52North's `CLA guidelines <https://52north.org/software/licensing/cla-guidelines/>`_ and, in particular, fill the 52North Contributor License Agreement.

Style guide
-----------

Follow the recommendations from `PEP 8 – Style Guide for Python Code <https://peps.python.org/pep-0008/>`_.

Docstring
---------

Docstrings should use the reStructuredText (reST) format. For details see, e.g., `this stackoverflow question <https://stackoverflow.com/questions/3898572/what-are-the-most-common-python-docstring-formats>`_, the `Sphinx + ReadTheDocs documentation <https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html>`_ or the example below.

Example:

.. code-block:: python

    This is a reStructuredText style.
    """
    :param param1: first param
    :type param1: type of first param
    :return: description of what is returned
    :rtype: type of return value
    :raises keyError: raises an exception
    """

Further reading:

 - `PEP 257 – Docstring Conventions <https://peps.python.org/pep-0257/>`_
 - `PEP 287 - reStructuredText Docstring Format <https://peps.python.org/pep-0287/>`_
