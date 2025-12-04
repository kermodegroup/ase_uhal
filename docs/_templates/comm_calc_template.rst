{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__
   .. automethod:: resample_committee
   .. automethod:: select_structure
   .. automethod:: sync
   .. automethod:: get_descriptor_energy
   .. automethod:: get_descriptor_forces
   .. automethod:: get_descriptor_stress
   .. automethod:: get_committee_energies
   .. automethod:: get_committee_forces
   .. automethod:: get_committee_stresses
   .. automethod:: get_bias_energy
   .. automethod:: get_bias_forces
   .. automethod:: get_bias_stress

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
