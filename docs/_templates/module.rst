{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:

{% block attributes %}
{%- if attributes %}
.. rubric:: {{ _('Module Attributes') }}

.. autosummary::
   :toctree:
   :template: attribute.rst
{% for item in attributes %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

{%- block classes %}
{%- if classes %}
.. rubric:: {{ _('Classes') }}

.. autosummary::
   :toctree:
   :template: class.rst
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

{%- block functions %}
{%- if functions %}
.. rubric:: {{ _('Functions') }}

.. autosummary::
   :toctree:
   :template: function.rst
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

{%- block exceptions %}
{%- if exceptions %}
.. rubric:: {{ _('Exceptions') }}

.. autosummary::
   :toctree:
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

{%- block modules %}
{%- if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: module.rst
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}
