# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'WeatherRoutingTool'
copyright = '2025, 52°North Spatial Information Research GmbH'
author = 'Katharina Demmich, Martin Pontius'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.graphviz'
]

# Configuration for inheritance diagrams
inheritance_graph_attrs = dict(rankdir="TB", size='"6.0, 8.0"',
                              fontsize=14, ratio='compress')

inheritance_node_attrs = dict(shape='box', fontsize=14, 
                             style='"rounded,filled"', 
                             fillcolor='white')

inheritance_edge_attrs = dict(penwidth=1.2, arrowhead='empty')

# Configuration for UML diagram output
graphviz_output_format = 'svg'

graphviz_dot_args = [
    "-Granksep=0.5",
    "-Gsplines=true", 
    "-Gnodesep=0.3",
    "-Gfontsize=12",
    "-Nfontsize=10",
    "-Nshape=box",
    "-Nstyle=rounded",
    "-Efontsize=9"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

pygments_style = 'sphinx'
