# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# 将项目的根目录添加到 sys.path
sys.path.insert(0, os.path.abspath('../../'))

project = 'pyRFM'
copyright = '2025, Yifei Sun'
author = 'Yifei Sun'
release = '0.1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_baseurl = 'https://ifaay.github.io/pyRFM/docs/'
html_static_path = ['_static']

base_url = "https://raw.githubusercontent.com/IFaay/pyRFM/master/docs/"

html_context = {
    "css_files": [
        f"{base_url}_static/pygments.css",
        f"{base_url}_static/css/theme.css",
    ],
    "script_files": [
        f"{base_url}_static/jquery.js",
        f"{base_url}_static/sphinx_highlight.js",
        f"{base_url}_static/doctools.js",
        f"{base_url}_static/js/theme.js",
    ]
}
