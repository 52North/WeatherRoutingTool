.. _contribution_guidelines:


Contribution Guidelines
=======================

Before getting involved please carefully read our `documentation for Open Source Software contributions <https://52north.org/software/contribute/>`_.
Be aware that we can only consider pull requests of authors who consider 52North's `CLA guidelines <https://52north.org/software/licensing/cla-guidelines/>`_ and, in particular, fill the 52North Contributor License Agreement.

General remarks
---------------

- Please do not ask if you can work on an issue. Contributions are welcome. Remember to read the `CLA guidelines <https://52north.org/software/licensing/cla-guidelines/>`_.
- Be aware that we do not assign issues to contributors we have not worked with yet. If this applies to you please do not ask to be assigned.

Commit messages
---------------

Commit messages follow the format ``<type> (<scope>): <short summary>``, e.g. ``fix (genetic): make speed spread for initial population respect boundaries``.
The repository provides a commit message template (``.gitmessage``) that lists the available types. Enable it for your local clone by running:

.. code-block:: bash

    git config commit.template .gitmessage
