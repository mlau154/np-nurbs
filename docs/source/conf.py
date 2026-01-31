import np_nurbs.version
import os

# Project Information
project = 'np-nurbs'
copyright = '2026, Matthew G. Lauer'
author = 'Matthew G. Lauer'

# Release Information
release = np_nurbs.version.get_major_project_version()
version = np_nurbs.version.__version__

# Sphinx Extensions
extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel',
    'sphinx_design',
]

# Allow Sphinx to get links to individual modules of external Python libraries
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'rust_nurbs': ('https://rust-nurbs.readthedocs.io/en/latest/', None),
}

# Theme
html_theme = 'pydata_sphinx_theme'

# Templates Path
templates_path = ['_templates']

# Static path
html_static_path = ['_static']

# Custom CSS file location
html_css_files = [
    'css/custom.css',
]

# Logo
html_logo = "_static/aerocaps_logo.png"

# Custom PyPI logo file
html_js_files = [
   "pypi-icon.js"
]

# Icon links
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mlau154/np-nurbs",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome"
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/rust-nurbs/",
            "icon": "fa-custom fa-pypi",
            "type": "fontawesome"
        }
   ]
}

