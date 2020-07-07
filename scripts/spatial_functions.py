#!/usr/bin/env python

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

'''
Created on 7/7/2019
@author: Neil Symington

Spatial functions used for various bits an pieces
'''

from scipy.spatial.ckdtree import cKDTree
import numpy as np

def depth_to_thickness(depth):
    """
    Function for calculating thickness from depth array
    :param depth: an array of depths
    :return:
    a flat array of thicknesses with the last entry being a null
    """
    # Create a new thickness array
    thickness = np.nan*np.ones(shape=depth.shape,
                               dtype=np.float)
    # Iterate through the depth array
    if len(depth.shape) == 1:
        thickness[0:-1] = depth[1:] - depth[:-1]
        return thickness

    elif len(depth.shape) == 2:
        thickness[:, 0:-1] = depth[:, 1:] - depth[:, :-1]
        return thickness
        
    elif len(depth.shape) == 3:

        thickness[:-1,:,:] = depth[1:,:, :] - depth[:-1,:, :]
        return thickness

def nearest_neighbours(points, coords, points_required = 1,
                       max_distance = 250.):
    """
    An implementation of nearest neaighbour for spatial data that uses kdtrees

    :param points: array of points to find the nearest neighbour for
    :param coords: coordinates of points
    :param points_required: number of points to return
    :param max_distance: maximum search radius
    :return:
    """
    if len(np.array(points).shape) == 1:
        points = np.array([points])

    # Initialise tree instance
    kdtree = cKDTree(data=coords)

    # iterate throught the points and find the nearest neighbour
    distances, indices = kdtree.query(points, k=points_required,
                                      distance_upper_bound=max_distance)

    # Mask out infitnite distances in indices to avoid confusion
    mask = np.isfinite(distances)

    if not np.all(mask):

        distances[~mask] = np.nan

    return distances, indices
