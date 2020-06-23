"""
Script to extract data from SMAP products.

Started: 6th January 2020
Author: G. Worrall
"""
import warnings
import numpy as np
import h5py
from datetime import datetime, timedelta


class RetrievalQualityError(Exception):
    """Raised when there is a problem with the SM data quality."""
    pass


class NoValueError(Exception):
    """Raised when product footprint doesn't cover AOI."""
    pass


def find_nearest_pixel(product, keys, coordinates):
    """Use the lat, long product metadata to find the relevant pixel index.

    Args:
        product (h5py._hl.dataset.Dataset): loaded SMAP data file.
        keys (dict): contains the formatted product keys for data access
        coordinates (list): of the format [lat, long] both floats.
    Returns:
        tuple: array index (2D if L3 data) of relevant nearest pixel.

    Raises:
        ValueError

    Note:
        Retrieval quality flag info:
            https://nsidc.org/data/spl3smp#retrieval-qual-flag
    """
    layer = keys['layer']
    level = keys['lvl']
    lat = keys['lat']
    lon = keys['lon']

    def get_nearest(array, value):
        # Get nearest value index.
        idx = (np.abs(array - value)).argmin()
        idx = np.unravel_index(idx, array.shape)
        return idx

    def _find_nearest_pixel_L2(coordinates, product, layer):
        # L2 product handler for eponymous function
        product_lats = product[layer][lat][:]
        product_longs = product[layer][lon][:]

        # Stack the two 1D arrays to find closest point
        lat_lon = np.column_stack((product_lats, product_longs))
        adjusted = lat_lon - np.array([coordinates[0], coordinates[1]])
        summed = np.sum(np.abs(adjusted), axis=1)
        idx = np.abs(summed).argmin()

        return (idx)

    def _find_nearest_pixel_L3(coordinates, product, layer):
        # L3 product handler for eponymous function.
        # Format keys for AM versus PM layer
        nonlocal lat, lon

        product_lats = product[layer][lat][:, :]
        product_longs = product[layer][lon][:, :]

        # All rows the same for lat and all cols the same for long
        x = get_nearest(product_lats, coordinates[0])[0]
        y = get_nearest(product_longs, coordinates[1])[1]

        return (x, y)

    if level == 'L1':
        return _find_nearest_pixel_L2(coordinates, product, layer)

    if level == 'L2':
        if 'km' in layer:  # then downscaled with Sentinel-1 and special case
            return _find_nearest_pixel_L3(coordinates, product, layer)
        return _find_nearest_pixel_L2(coordinates, product, layer)

    if level == 'L3':
        return _find_nearest_pixel_L3(coordinates, product, layer)

    raise ValueError("Layer parameter must be 'L2' or 'L3'.")


def check_retrieval_quality(product, keys, index, raise_error):
    """Check the retrieval quality flag for a given index on a SMAP product.

    Args:
        product (h5py.File): SMAP product of concern
        keys (dict): contains the formatted product keys for data access
        index (tuple): (x, y) or (x) (L3/L2 data) array index
        raise_error (bool): If False, does not raise RetrievalQualityError
            for bad data.

    Returns:
        int: Returns the retrieval quality flag.

    Raises:
        RetrievalQualityError: If retrieval wasn't succesful.

    Note:
        Retrieval flags:
            https://nsidc.org/data/SPL3SMP/versions/6#retrieval-qual-flag
    """
    retrieval_errors = {'1': 'Soil moisture retrieval has uncertain quality.',
                        '5': 'Soil moisture retrieval was not successful.',
                        '7': 'Soil moisture retrieval was not successful.',
                        '8': 'Freeze/thaw state retrieval was not successful.',
                        '9': 'Freeze/thaw state retrieval was not successful'
                             'and soil moisture retrieval quality uncertain.',
                        '13': 'Soil moisture retrieval was not successful.',
                        '15': 'Soil moisture retrieval was not successful.'}

    layer = keys['layer']
    qual = keys['flag']

    flag = product[layer][qual][index]

    if 'sm' in keys.keys():  # SM flags
        if not raise_error:
            return flag
        if flag != 0 and flag != 8:  # 8 is freezethaw state only
            try:
                msg = retrieval_errors[str(flag)]
                if flag not in [1, 9]:  # Quality uncertain flag.
                    raise RetrievalQualityError(msg)
                # Only warn if quality uncertain, otherwise raise error.
                warnings.warn(msg)
                return flag
            except KeyError:
                raise RetrievalQualityError(
                    'Unknown retrieval_quality_flag for nearest pixel.'
                ) from None

    if 'tb' in keys.keys():
        if flag != 0:
            raise RetrievalQualityError("TB retrieval not marked as having"
                                        " acceptable quality.")

    return flag


def get_SM_key_terms(product, layer):
    """Defines and formats the key terms for SM product data access.

    Args:
        product (h5py.File): the SMAP product
        layer (str): the string originally provided by the user.

    Returns:
        dict: formatted strings for data access
    """
    # Define key terms
    keys = {'sm': 'soil_moisture',
            'time': 'tb_time_utc',
            'lvl': product['Metadata']['ProductSpecificationDocument'].attrs[
                'SMAPShortName'][:2].decode('utf-8'),
            'lat': 'latitude',
            'lon': 'longitude',
            'flag': 'retrieval_qual_flag',
            'layer': 'Soil_Moisture_Retrieval_Data'}

    # Format key terms
    if layer:
        for key in keys.keys():
            if key == 'lvl':  # don't modify lvl
                continue
            keys[key] = keys[key] + '_' + layer.lower()

    return keys


def _get_TB_key_terms(product, look_direction, polarization):
    """Defines and formats the key terms for SM product data access.

    Args:
        product (h5py.File): the SMAP product
        layer (str): the string originally provided by the user.

    Returns:
        dict: formatted strings for data access
    """
    # Define key terms
    keys = {'tb': 'cell_tb_{}_surface_corrected_{}'.format(polarization,
                                                           look_direction),
            'time': 'cell_tb_time_utc_{}'.format(look_direction),
            'lvl': product['Metadata']['ProductSpecificationDocument'].attrs[
                'SMAPShortName'][:2].decode('utf-8'),
            'lat': 'cell_lat_centroid_{}'.format(look_direction),
            'lon': 'cell_lon_centroid_{}'.format(look_direction),
            'flag': 'cell_tb_qual_flag_{}_{}'.format(polarization,
                                                     look_direction),
            'layer': 'Global_Projection'}

    return keys


def get_SMAP_SM(product_path, coordinates, layer=False, raise_error=True):
    """Extract SM data for a given set of coordinates from a SMAP product.

    Args:
        product_path (str): string of path to SMAP product
        coordinates (tuple): coordinates of point in the form (lat, long).
        layer (str, optional): 'AM' or 'PM' (L3) products / '1km' or '3km'
            (L2 Sentinel-1) products.
        raise_error (bool): If false, just returns uncertain / invalid quality
            flag rather than raising a RetrievalQualityError

    Returns:
        tuple: of the form (SM_value (float), retrieval_quality_flag (int),
            capture_time (datetime.datetime))"""

    f = h5py.File(product_path, 'r')

    # Value checks
    if layer and layer.lower() not in ['am', 'pm', '1km', '3km']:
        raise ValueError("Layer can only be 'AM', 'PM',"
                         " '1km', or '3km'.")

    keys = get_SM_key_terms(f, layer)

    layer = keys['layer']
    sm = keys['sm']
    time = keys['time']

    # Get index of pixel containing coordinate point
    index = find_nearest_pixel(f, keys, coordinates)
    # Get SM value
    sm_value = f[layer][sm][index]

    # Get retrieval quality flag
    flag = check_retrieval_quality(f, keys, index, raise_error)

    # No value placeholder is -9999.0
    if sm_value < 0 and raise_error:
        raise NoValueError("Product footprint does not cover area of"
                           " interest.\n HINT: Check your coordinates order."
                           " This script requires them in [lat, lon] form.")

    # Get capture time - discard milliseconds as sometimes timestring
    # leaves milliseconds as *** which does not work with datetime formatting
    if 'km' in layer.lower():
        # Sentinel enchanced products go from J2000 epoch
        seconds = f[layer]['spacecraft_overpass_time_seconds_3km'][index]
        capture_time = datetime(2000, 1, 1, 11, 58, 55, 816) + timedelta(
            seconds=seconds)
    else:
        capture_time = datetime.strptime(
            f[layer][time][index].decode('utf-8')[:-5],
            '%Y-%m-%dT%H:%M:%S')

    return (sm_value, flag, capture_time)


def get_SMAP_TB(product_path, coordinates, look_direction, polarization):
    """Extract TB data for a given set of coordinates from a SMAP product.

    Args:
        product_path (str): string of path to SMAP product
        coordinates (tuple): coordinates of point in the form (lat, long).
        look_direction (str): 'fore' or 'aft' look direction
        polarization (str): 'v' or 'h' polarization

    Returns:
        tuple: of the form (tb_value (float), retrieval_quality_flag (int),
            capture_time (datetime.datetime))

    Note: Extracts surfaced corrected TB from the product"""

    f = h5py.File(product_path, 'r')
    keys = _get_TB_key_terms(f, look_direction, polarization)
    index = find_nearest_pixel(f, keys, coordinates)
    tb_value = f['Global_Projection'][keys['tb']][index]

    if tb_value < 0:
        raise NoValueError("Product footprint does not cover area of"
                           " interest.")
    capture_time = datetime.strptime(
        f[keys['layer']][keys['time']][index].decode('utf-8')[:-5],
        '%Y-%m-%dT%H:%M:%S')

    flag = check_retrieval_quality(f, keys, index)

    return (tb_value, flag, capture_time)


def get_pixel_extents(product, keys, index):
    """Calculate boundary corners of pixel from neighbouring pixel centroids.
    Args:
        product (h5py.File): SMAP product of concern
        keys (dict): contains the formatted product keys for data access
        index (tuple): (x, y) or (x) (L3/L2 data) array index

    Returns:
        list: containing coordinates of boundary corners of pixel


    NOTE: Calculates the mid point between diagonally adjacent pixels to
        calculate pixel boundary corners.
        For edge case where pixel is edge pixel, then estimates the edge
        distance from the other side.
    """
    raise NotImplementedError
