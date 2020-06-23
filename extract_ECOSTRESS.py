"""
Script to extract data from ECOSTRESS products.

Started: 24th May 2020
Author: G. Worrall
"""
import h5py
import pyproj
import numpy as np
from pyresample import geometry as geom
from pyresample import kd_tree as kdt
from osgeo import gdal, gdal_array, osr
from pathlib import Path


def process_ECOSTRESS_ET(product_path, geofile_path):
    """Process ET data file into .tif file for a given ECOSTRESS product.

    Based on USGS tutorial at:
        https://lpdaac.usgs.gov/resources/e-learning/
        working-ecostress-evapotranspiration-data/

    Args:
        product_path (str): string of path to ECOSTRESS product
        geofile_path (str): path to L1B ECOSTRESS geolocation file
        coordinates (tuple): coordinates of area in the form (lat, long).

    Returns:
        (ET_file_path, ETUncert_file_path): paths to the processed files

        """
    file_names = ['ETinst', 'ETinstUncertainty']
    if Path('{}_{}.tif'.format(product_path[:-3], file_names[0])).exists():
        return ['{}_{}.tif'.format(product_path[:-3], x) for x
                in file_names]  # already processed

    f = h5py.File(product_path, 'r')

    # Get all SDS layers in dataset
    eco_objs = []
    f.visit(eco_objs.append)
    ecoSDS = [str(obj) for obj in eco_objs if isinstance(f[obj], h5py.Dataset)]

    # subset so we just have ET and ET uncertainty
    sds = ['ETinst', 'ETinstUncertainty']
    ecoSDS = [dataset for dataset in ecoSDS if dataset.endswith(tuple(sds))]

    # open the geo file
    g = h5py.File(geofile_path, 'r')
    geo_objs = []
    g.visit(geo_objs.append)

    # Search for lat/lon SDS inside data file
    latSD = [str(obj) for obj in geo_objs
             if isinstance(g[obj], h5py.Dataset) and '/latitude' in obj]
    lonSD = [str(obj) for obj in geo_objs
             if isinstance(g[obj], h5py.Dataset) and '/longitude' in obj]

    # Open SDS as arrays
    lat = g[latSD[0]][()].astype(np.float)
    lon = g[lonSD[0]][()].astype(np.float)

    # Set swath definition from lat/lon arrays
    swathDef = geom.SwathDefinition(lons=lon, lats=lat)

    # Define the lat/lon for the middle of the swath
    mid = [int(lat.shape[1] / 2) - 1, int(lat.shape[0] / 2) - 1]
    midLat, midLon = lat[mid[0]][mid[1]], lon[mid[0]][mid[1]]

    # Define AEQD projection centered at swath center
    epsgConvert = pyproj.Proj("+proj=aeqd +lat_0={} +lon_0={}".format(
        midLat, midLon))

    # Use info from AEQD projection bbox to calculate output cols/rows/pixel
    # size
    llLon, llLat = epsgConvert(np.min(lon), np.min(lat), inverse=False)
    urLon, urLat = epsgConvert(np.max(lon), np.max(lat), inverse=False)
    areaExtent = (llLon, llLat, urLon, urLat)
    cols = int(round((areaExtent[2] - areaExtent[0]) / 70))  # 70 m pixel size
    rows = int(round((areaExtent[3] - areaExtent[1]) / 70))
    # Define Geographic projection
    epsg, proj, pName = '4326', 'longlat', 'Geographic'

    # Define bounding box of swath
    llLon, llLat, urLon, urLat = (np.min(lon), np.min(lat),
                                  np.max(lon), np.max(lat))
    areaExtent = (llLon, llLat, urLon, urLat)

    # Create area definition with estimated number of columns and rows
    projDict = {'proj': proj, 'datum': 'WGS84'}
    areaDef = geom.AreaDefinition(
        epsg, pName, proj, projDict, cols, rows, areaExtent)

    # Square pixels and calculate output cols/rows
    ps = np.min([areaDef.pixel_size_x, areaDef.pixel_size_y])
    cols = int(round((areaExtent[2] - areaExtent[0]) / ps))
    rows = int(round((areaExtent[3] - areaExtent[1]) / ps))

    # Set up a new Geographic area definition with the refined cols/rows
    areaDef = geom.AreaDefinition(
        epsg, pName, proj, projDict, cols, rows, areaExtent)

    # Get arrays with information about the nearest neighbor to each grid point
    index, outdex, indexArr, distArr = kdt.get_neighbour_info(
        swathDef, areaDef, 210, neighbours=1)

    # Read in ETinst and print out SDS attributes
    s = ecoSDS[0]
    ecoSD = f[s][()]

    # Read SDS attributes and define fill value,
    # add offset, and scale factor if available
    try:
        fv = int(f[s].attrs['_FillValue'])
    except KeyError:
        fv = None
    except ValueError:
        fv = f[s].attrs['_FillValue'][0]
    try:
        sf = f[s].attrs['_Scale'][0]
    except IndexError:
        sf = 1
    try:
        add_off = f[s].attrs['_Offset'][0]
    except IndexError:
        add_off = 0

    fv = -9999

    # Perform K-D Tree nearest neighbor resampling (swath 2 grid conversion)
    ETgeo = kdt.get_sample_from_neighbour_info(
        'nn', areaDef.shape, ecoSD, index, outdex, indexArr, fill_value=fv)

    # Define the geotransform
    gt = [areaDef.area_extent[0], ps, 0, areaDef.area_extent[3], 0, -ps]

    ETgeo = ETgeo * sf + add_off            # Apply Scale Factor and Add Offset
    ETgeo[ETgeo == fv * sf + add_off] = fv

    # Repeat the above for the uncertainty file
    s = ecoSDS[1]
    ecoSD = f[s][()]
    try:
        fv = int(f[s].attrs['_FillValue'])
    except KeyError:
        fv = None
    except ValueError:
        fv = f[s].attrs['_FillValue'][0]
    try:
        sf = f[s].attrs['_Scale'][0]
    except IndexError:
        sf = 1
    try:
        add_off = f[s].attrs['_Offset'][0]
    except IndexError:
        add_off = 0
    fv = -9999
    UNgeo = kdt.get_sample_from_neighbour_info(
        'nn', areaDef.shape, ecoSD, index, outdex, indexArr, fill_value=fv)
    UNgeo = UNgeo * sf + add_off
    UNgeo[UNgeo == fv * sf + add_off] = fv

    # Set up dictionary of arrays to export
    outFiles = {'ETinst': ETgeo, 'ETinstUncertainty': UNgeo}

    processed_products = []
    # Loop through each item in dictionary created above
    for f in outFiles:

        # Set up output name
        outName = '{}_{}.tif'.format(product_path[:-3], f)
        print("output file:\n{}\n".format(outName))

        # Get driver, specify dimensions, define and set output geotransform
        height, width = outFiles[f].shape
        driv = gdal.GetDriverByName('GTiff')
        dataType = gdal_array.NumericTypeCodeToGDALTypeCode(outFiles[f].dtype)
        d = driv.Create(outName, width, height, 1, dataType)
        d.SetGeoTransform(gt)

        # Create and set output projection, write output array data
        # Define target SRS
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(epsg))
        d.SetProjection(srs.ExportToWkt())
        srs.ExportToWkt()

        # Write array to band
        band = d.GetRasterBand(1)
        band.WriteArray(outFiles[f])

        # Define fill value if it exists, if not, set to mask fill value
        if fv is not None and fv != 'NaN':
            band.SetNoDataValue(fv)
        else:
            try:
                band.SetNoDataValue(outFiles[f].fill_value)
            except AttributeError:
                pass
        band.FlushCache()
        d, band = None, None

        processed_products.append(outName)

    return processed_products
