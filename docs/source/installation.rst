.. _installation:

Installation
============

Steps:

1. Clone the repository:

.. code-block:: shell

    git clone https://github.com/52North/WeatherRoutingTool.git

2. Change to the folder:

.. code-block:: shell

    cd WeatherRoutingTool

3. [recommended] Create and activate a virtual environment, e.g.

.. code-block:: shell

    python3 -m venv "venv"
    source venv/bin/activate

4. Install the WRT:

    4.1. In normal mode

    .. code-block:: shell

        pip install .

    4.2. In editable mode (recommended for development)

    .. code-block:: shell

        pip install -e .

**Power/fuel consumption framework**

In order to get high-quality results, a suitable power/fuel modelling framework should be used as it is the core of any weather routing optimization. Please check the respective section of our documentation for more information as the installation of dedicated software might be necessary for your application. 

What to do after installation
=============================

After successfully installing WeatherRoutingTool, you can verify that the installation works correctly.

1. Activate your virtual environment.
2. Run the help command:

::

   python main.py --help

If the help message is displayed without errors, the installation was successful.

You can also run the test suite:

::

   pytest

If all tests pass, your setup is working correctly.

Why use a virtual environment?
------------------------------

A virtual environment isolates project dependencies from your global Python installation. 
This prevents version conflicts between different Python projects and ensures reproducibility.