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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import cycler
import matplotlib as mpl

project = "splinebox"
copyright = "2024, Florian Aymanns, Edward Ando, Virginie Uhlmann"  # noqa: A001
author = "Florian Aymanns, Edward Ando, Virginie Uhlmann"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_title = "Splinebox Documentation"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_sourcelink_suffix = ""
html_short_title = "SplineBox"

html_theme_options = {
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/EPFL-Center-for-Imaging/splinebox",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/splinebox/",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "Contact",
            "url": "mailto: florian.aymanns@epfl.ch",
            "icon": "fa-brands fa-telegram",
        },
        {
            "name": "EPFL Center for Imaging",
            "url": "https://imaging.epfl.ch/",
            "icon": "_static/imaging.png",
            "type": "local",
        },
    ],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navbar_align": "content",  # [left, content, right] For testing that the navbar items align properly
    "navbar_center": ["navbar-nav"],
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc", "searchbox", "edit-this-page", "sourcelink"],
    "footer_start": ["copyright"],
    "pygment_light_style": "tango",
}

html_context = {
    "github_user": "EPFL-Center-for-Imaging",
    "github_repo": "splinebox",
    "github_version": "main",
    "doc_path": "doc",
    "default_mode": "auto",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]


def reset_mpl(gallery_conf, fname):
    mpl.rcParams["axes.prop_cycle"] = cycler.cycler(color=["#228b18"])


# Example gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "matplotlib_animations": True,
    "reset_modules": (reset_mpl,),
}

# Matplotlib sphinxext configuration
plot_html_show_formats = False
plot_html_show_source_link = False
plot_formats = [
    ("png", 500),
]
plot_rcparams = {"axes.prop_cycle": cycler.cycler(color=["#228b18"])}
