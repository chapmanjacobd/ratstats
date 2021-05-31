"""
python raster_stats_vector_zones.py grid.tif liechtenstein_grass.sqlite

MIT License
Based on https://github.com/pcjericks/py-gdalogr-cookbook/blob/master/raster_layers.rst#calculate-zonal-statistics
"""

import sys
import ogr
import numpy
import gdal
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas
import os


def zonal_stats(raster, lyr, FID):

    # Get raster georeference info
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]

    feat = lyr.GetFeature(FID)

    # Get extent of feat
    geom = feat.GetGeometryRef()
    xmin, xmax, ymin, ymax = geom.GetEnvelope()

    # Specify offset and rows and columns to read
    xoff = int((xmin - xOrigin) / pixelWidth)
    yoff = int((yOrigin - ymax) / pixelWidth)
    xcount = int((xmax - xmin) / pixelWidth) + 1
    ycount = int((ymax - ymin) / pixelWidth) + 1

    dataraster = raster.ReadAsArray(xoff, yoff, xcount, ycount)

    if dataraster is None:
        return dict(
            id=FID,
            min=None,
            median=None,
            max=None,
        )

    return dict(
        id=FID,
        min=dataraster.min(),
        median=numpy.median(dataraster).astype(int),
        max=dataraster.max(),
        # mean=dataraster.mean().astype(int),
        # std=dataraster.std().astype(int),
        # var=numpy.var(dataraster).astype(int),
    )


def raster_stats_vector_zones(input_zone_polygon, input_value_raster):
    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()
    print("Processing", lyr.GetFeatureCount(), "features")
    raster = gdal.Open(input_value_raster)

    # ogr.Open is very expensive so let's leave this at n_jobs=1
    stats = Parallel(n_jobs=1)(
        delayed(zonal_stats)(raster, lyr, FID)
        for FID in tqdm(range(1, lyr.GetFeatureCount() + 1))
    )

    pandas.DataFrame.from_records(stats).to_csv(
        os.path.splitext(input_value_raster)[0]
        + os.path.splitext(input_zone_polygon)[0]
        + ".csv",
        index=False,
    )


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "[ ERROR ] you must supply two arguments: input-zone-shapefile-name.shp input-value-raster-name.tif "
        )
        sys.exit(1)

    raster_stats_vector_zones(sys.argv[1], sys.argv[2])
