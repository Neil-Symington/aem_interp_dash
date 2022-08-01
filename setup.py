#===============================================================================
#    Copyright 2017 Geoscience Australia
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#===============================================================================

"""
Created on 16/12/2020
@author: Neil Symington
"""
from distutils.core import setup

setup(name='garjmcmctdem_utils',
      version='0.1',
      description='Utilities for AEM interpretation using dash',
      url='https://github.com/Neil-Symington/aem_interp_dash',
      author='Neil Symington',
      author_email='neil.symington@ga.gov.au',
      requires = [
                 'matplotlib',
                 'netcdf4',
                 'numpy',
                 'pandas',
                 'geopandas',
                 'xarray',
                 'dash',
                 'plotly',
                 'sklearn',
                 'pyyaml',
                 'affine',
                 'shapely',
                 'scipy'
              ],
      packages=['garjmcmctdem_utils'],
      license='Apache License Version 2.0')