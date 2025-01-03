pyRFM: A Python Package for Random Feature Method (RFM)
========================================================

pyRFM is a Python package designed to implement the Random Feature Method (RFM) for solving partial differential equations (PDEs).

Contents:

.. toctree::
   :maxdepth: 2
   :caption: Modules

   core
   utils
   geometry
   voronoi

Version: 0.1.2

Quick Install
-------------

To install pyRFM, use one of the following methods:

.. code-block:: bash

   pip install git+https://github.com/IFaay/pyRFM.git

Or clone the repository and install locally:

.. code-block:: bash

   git clone https://github.com/IFaay/pyRFM.git
   cd pyRFM
   pip install .

Update
------

To update to the latest version, use:

.. code-block:: bash

   pip install --upgrade git+https://github.com/IFaay/pyRFM.git

And ensure that you re-download or pull the latest source code.

Remark
------

All examples run successfully on a host equipped with:
- **8GB of GPU memory**
- **16GB of RAM**

Example scripts are located in the `examples <https://github.com/IFaay/pyRFM/tree/master/examples>`_ folder.

Reference
---------

The following references provide more context and details about the Random Feature Method:

1. J. Chen, X. Chi, W. E, and Z. Yang, “Bridging traditional and machine learning-based algorithms for solving PDEs:
   The random feature method,” *Journal of Machine Learning*, vol. 1, no. 3, pp. 268–298, 2022.
   doi: `10.4208/jml.220726 <https://doi.org/10.4208/jml.220726>`_.

2. J. Chen, W. E, and Y. Luo, “The random feature method for time-dependent problems,” *East Asian Journal on Applied Mathematics*,
   vol. 13, no. 3, pp. 435–463, 2023.
   doi: `10.4208/eajam.2023-065.050423 <https://doi.org/10.4208/eajam.2023-065.050423>`_.

3. J. Chen, W. E, and Y. Sun, “Optimization of Random Feature Method in the High-Precision Regime,” *Communications in Applied Mathematics and Computation*,
   Mar. 2024.
   doi: `10.1007/s42967-024-00389-8 <https://doi.org/10.1007/s42967-024-00389-8>`_.