import numpy as np
from scipy.signal import argrelmax
import requests



__author__ = 'Max Floettmann'

"""
An experimental module to estimate positions of bus stops from crowd data.
"""

import geojson
import geopandas
import dalladalla.weightedkde as kde

class Estimator:
    """
    Holds data needed for estimation of bus stops and provides estimation methods.
    """
    def __init__(self, input_points:geopandas.geoseries.GeoSeries=None, weights:np.ndarray=None):
        """

        :param input_points: Geometries from a Geodataframe containing points that can be used for estimation.
        :param weights: vector of weights for each point
        """
        self._stop_prob = None
        self.points = input_points
        self.weights = weights

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value:geopandas.GeoDataFrame):
        self.__p_coordinates = np.array([p.xy for p in value.geometry]).reshape((value.shape[0],2)).T
        self._points = value
        self.__boundaries = np.array([self.__p_coordinates.min(axis=1),
                                      self.__p_coordinates.max(axis=1)]).flatten()[[0,2,1,3]]
        self._stop_prob = None

    @property
    def weights(self) -> np.array:
        return self._weights

    @weights.setter
    def weights(self, value: np.ndarray):
        self._weights = value
        self._stop_prob = None

    @property
    def stop_prob(self) -> np.ndarray:
        """
        An array of the estimated probability of a bus stop.
        """
        return self._stop_prob

    @property
    def boundaries(self):
        return self.__boundaries

    def estimate_stops(self, weighted:bool=True, snap_to_street:bool=False,
                       resolution:int=100, kernel_width:float=0.1) -> np.array:
        """
        Estimate the position of bus stops using a kernel density estimator. The kernel is estimated using the given
        points and then evaluated on a grid of size resolution x resolution. The local maxima in this grid are then used as an
        estimation for bus stops.

            :param weighted: should the weights be used for estimation.
            :param snap_to_street: Should the points be moved to the nearest street.
            :param resolution: width and heigth of raster
            :param kernel_width: width of the kernel we use for estimation
        """
        x = np.linspace(self.__boundaries[0], self.__boundaries[1], resolution)
        y = np.linspace(self.__boundaries[2], self.__boundaries[3], resolution)
        xx, yy = np.meshgrid(x, y)
        if not self._stop_prob:
            # Evaluate the kde on a grid
            if weighted:
                pdf = kde.gaussian_kde(self.__p_coordinates, weights=self.weights, bw_method=kernel_width)
            else:
                pdf = kde.gaussian_kde(self.points, bw_method=kernel_width)
            stop_prob = pdf((np.ravel(xx), np.ravel(yy)))
            stop_prob = np.reshape(stop_prob, xx.shape)
            self._stop_prob = stop_prob
        maxima_x = argrelmax(self._stop_prob, axis=1)
        maxima_y = argrelmax(self._stop_prob, axis=0)
        out_x = np.zeros_like(self._stop_prob)
        out_x[maxima_x[0], maxima_x[1]] = 1
        out_y = np.zeros_like(self._stop_prob)
        out_y[maxima_y[0], maxima_y[1]] = 1
        out = out_x + out_y
        stop_points = np.array([xx[out == 2.], yy[out == 2]])
        if snap_to_street:
            stop_points = np.apply_along_axis(nearest_street, 1, stop_points.T).T
        point_list = stop_points.tolist()
        return list(zip(point_list[1], point_list[0]))

    def route_dist(self, routes: geopandas.geoseries.GeoSeries) -> list:
        """
        Calculate the distance of each point to the closest bus route.

        :param routes: Given geometries of bus routes
        """
        min_dist = []
        for point in self.points:
            min_dist.append(min([point.distance(r) for r in routes]))
        return min_dist



def nearest_street(coordinates, host='http://router.project-osrm.org'):
    # FIXME: API does not seem to do what I expected, check
    """
    Return the coordinates of the nearest street for any coordinates using OSRM.
    Not working yet. I understood the API would return the snapped coordinate, but it does not:
    https://github.com/Project-OSRM/osrm-backend/wiki/Server-api

    :type coordinates: list(float)
    """
    try:
        url = '{}/nearest?loc={},{}'.format(host, coordinates[0], coordinates[1])
        response = requests.get(url)
        return response.json()['mapped_coordinate']
    except Exception as e:
        print("OSRM error: {}".format(e))
        return 0