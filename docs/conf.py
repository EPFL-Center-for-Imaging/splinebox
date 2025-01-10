import datetime
import os
import re
import sys

import cycler
import matplotlib as mpl
import numpydoc.docscrape as np_docscrape
import pyvista
import splinebox
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# -- Project information -----------------------------------------------------
project = "SplineBox"
copyright = f"2024-{datetime.date.today().year}, Florian Aymanns, Edward Ando, Virginie Uhlmann"  # noqa: A001
author = "Florian Aymanns, Edward Ando, Virginie Uhlmann"

version = re.sub(r"\.dev.*$", r".dev", splinebox.__version__)
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_gallery.gen_gallery",
    # "pyvista.ext.plot_directive",
    "pyvista.ext.viewer_directive",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_title = "SplineBox Documentation"
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
    "navbar_align": "content",
    "navbar_center": ["navbar-nav"],
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
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

# sphinx-copybutton configurations
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.{3,}: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Options for example gallery --------------------------------------------------------


def reset_mpl(gallery_conf, fname):
    mpl.rcParams["axes.prop_cycle"] = cycler.cycler(color=["#228b18"])


sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "_auto_examples",  # path to where to save gallery generated output
    "default_thumb_file": "./_static/favicon.png",
    "matplotlib_animations": True,
    "reset_modules": (reset_mpl,),
    # Remove sphinx configuration comments from code blocks
    "remove_config_comments": True,
    # directory where function granular galleries are stored
    # "backreferences_dir": None,
    # Modules for which function level galleries are created.
    "doc_module": "pyvista",
    "image_scrapers": (DynamicScraper(), "matplotlib"),
    "first_notebook_cell": ("%matplotlib inline\nfrom pyvista import set_plot_theme\nset_plot_theme('document')\n"),
    "reset_modules_order": "both",
}

# -- Options for matplotlib -------------------------------------------------------------

plot_include_source = True
plot_html_show_formats = False
plot_html_show_source_link = False
plot_formats = [
    ("png", 500),
]
plot_rcparams = {"axes.prop_cycle": cycler.cycler(color=["#228b18"])}

# -- Options for autosectionlabel -------------------------------------------------------

# This is useful to prevent duplicate section label warnings
autosectionlabel_prefix_document = True

# -- Options for autosummary ------------------------------------------------------------

# Tell autosummary to generate the rst files
# for the items in the summary.
autosummary_generate = True

# -- Options for autodoc ----------------------------------------------------------------

autodoc_default_options = {
    "inherited-members": None,
}
autodoc_typehints = "none"

# -- Options for numpydoc ---------------------------------------------------------------

numpydoc_use_plots = True
np_docscrape.ClassDoc.extra_public_methods = ["__call__"]  # should match custom-autosummary-class-template.rst

# -- Options for pyvista plots ----------------------------------------------------------

# Ensure that offscreen rendering is used for docs generation
pyvista.OFF_SCREEN = True  # Not necessary - simply an insurance policy
# Preferred plotting style for documentation
pyvista.set_plot_theme("document")

# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
os.environ["PYVISTA_BUILDING_GALLERY"] = "true"
