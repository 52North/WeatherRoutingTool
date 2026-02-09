.. _installation:

Installation
============
Requirements : 
Virtual Environment(venv): A virtual environment is an isolated directory on a computer that contains a specific Python installation, along with its own libraries, scripts, and dependencies. It acts as a sandbox, allowing developers to manage project-specificdependencies without conflicting with global system packages or other projects. 

1. A virtual environment is required in Python to isolate project dependencies, preventing version conflicts between libraries.

2. It creates an independent directory containing its own Python executable and packages, allowing different projects to use different versions of the same library. It is essential for maintaining clean, reproducible development environments. 


Steps:

1. Clone the repository:

.. code-block:: shell/terminal

    git clone https://github.com/52North/WeatherRoutingTool.git

2. Change to the folder:

.. code-block::   shell/terminal

    cd WeatherRoutingTool

3. [recommended] Create and activate a virtual environment, e.g.

.. code-block::   shell/terminal

    python3 -m venv "venv"/py -m venv "venv"

    for mac:

    source venv/bin/activate

    for windows:

    .\venv\Scripts\Activate.ps1

4. Install the WRT:

    4.1. In normal mode

    .. code-block::  shell/terminal

        pip install . && pip install --no-deps -r requirements-without-deps.txt

    4.2. In editable mode (recommended for development)

    .. code-block::  shell/terminal

        pip install -e . && pip install --no-deps -r requirements-without-deps.txt

The part ``pip install --no-deps -r requirements-without-deps.txt`` is necessary because of a dependency issue (see `issue 8 <https://github.com/52North/WeatherRoutingTool/issues/8>`_). We might implement a different solution in the future making the installation easier/cleaner.
 
 WHAT TO DO AFTER INSTALLATION ? 

After you have successfully installed the tool, the next step is to run a test simulation to see it in action.
The tool needs specific weather data (GRIB files) and boat polar files to work, the absolute best way to start is by using the Sandbox.

Steps:

1. Get the Sample Data (The Sandbox):

   Clone the sandbox :

   .. code-block:: In integrated terminal
         cd ..

         git clone https://github.com/52North/WRT-sandbox.git

2. Change to the folder:

.. code-block::  shell/terminal

         cd WeatherRoutingTool

**Configure the tool** 
  Open "Configuration/config.template.json" and ensure the paths are correct for your Operating System

3. Run Your first "Weather Route"(make sure (venv )is still active)
 Set your ``PYTHONPATH`` to the root of the ``WeatherRoutingTool`` directory and execute the run script:

.. code-block::  shell/terminal

         $env:PYTHONPATH = "path/to/WeatherRoutingTool"
         python run_WRT.py
         

**Power/fuel consumption framework**

In order to get high-quality results, a suitable power/fuel modelling framework should be used as it is the core of any weather routing optimization. Please check the respective section of our documentation for more information as the installation of dedicated software might be necessary for your application. 
