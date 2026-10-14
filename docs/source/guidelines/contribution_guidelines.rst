.. _contribution_guidelines:


Contribution Guidelines
=======================

Before getting involved please carefully read our `documentation for Open Source Software contributions <https://52north.org/software/contribute/>`_.
Be aware that we can only consider pull requests of authors who consider 52North's `CLA guidelines <https://52north.org/software/licensing/cla-guidelines/>`_ and, in particular, fill the 52North Contributor License Agreement.

General remarks
---------------
We welcome contributions that make meaningful improvements to the codebase, such as bug fixes, new features, performance improvements, or substantial refactoring.
Please note that we do **not** accept pull requests that only consist of:

    - Documentation changes (including README edits, typo fixes, or comment updates)
    - Formatting or whitespace adjustments
    - Other trivial or cosmetic changes
   
Pull requests of this kind will be closed without review. If you've spotted an error in the documentation, please open an issue and we'll address it ourselves. Documentation updates are, of course, welcome as part of a larger code contribution when they describe the changes you've made.

Please do not ask if you can work on an issue: contributions are welcome! However, remember to read the `CLA guidelines <https://52north.org/software/licensing/cla-guidelines/>`_.
Be aware that we do not assign issues to contributors we have not worked with yet. If this applies to you please do not ask to be assigned.

Commit messages
---------------

Commit messages follow the format ``<type> (<scope>): <short summary>``, e.g. ``fix (genetic): make speed spread for initial population respect boundaries``.
The repository provides a commit message template (``.gitmessage``) that lists the available types. Enable it for your local clone by running:

.. code-block:: bash

    git config commit.template .gitmessage
