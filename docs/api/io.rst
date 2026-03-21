IO utilities
============

.. automodule:: pasted._io
   :members:
   :undoc-members: False

.. note::

   **Extended XYZ comment-line format.**

   The comment line written by PASTED follows this structure::

       sample=N mode=M charge=+Q mult=M comp=[El1:n1,El2:n2,...]  KEY1=V1  KEY2=V2  ...

   ``comp=`` encodes the composition as a sorted comma-separated list of
   ``Element:count`` pairs.  All metric keys from
   :data:`~pasted._atoms.ALL_METRICS` appear in order; ``nan`` is written
   for any metric that could not be computed.  Metric values are formatted
   to 4 decimal places.

   :func:`~pasted._io.parse_xyz` extracts ``charge``, ``mult``, and any
   ``KEY=FLOAT`` tokens from this line.  Unknown keys are silently ignored,
   making the format forward-compatible with future metric additions.

.. rubric:: High-level helpers

For most use-cases the higher-level methods on
:class:`~pasted._generator.Structure` are more convenient than calling
:func:`~pasted._io.format_xyz` and :func:`~pasted._io.parse_xyz` directly:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method / function
     - Description
   * - :meth:`Structure.to_xyz() <pasted._generator.Structure.to_xyz>`
     - Serialise one structure to an extended XYZ string in memory.
   * - :meth:`Structure.write_xyz() <pasted._generator.Structure.write_xyz>`
     - Write or append one frame to a file.
   * - :func:`pasted._generator.read_xyz`
     - Load all frames from a file or raw string and return
       ``list[Structure]`` with metrics recomputed by default.
