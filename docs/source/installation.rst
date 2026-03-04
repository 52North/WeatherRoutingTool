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
