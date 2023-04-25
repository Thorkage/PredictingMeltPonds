#! /usr/bin/env python

import gdal, gdalconst,osr
import numpy as np
from scipy.cluster.vq import *
import sys
import os

###N. Neckel 2020

def ImportGeoTiff(GTIFFref):
	match_filename = GTIFFref
	match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
	data = match_ds.ReadAsArray()
	match_proj = match_ds.GetProjection()
	match_geotrans = match_ds.GetGeoTransform()
	width = match_ds.RasterXSize
	height = match_ds.RasterYSize
	return data,width,height,match_geotrans,match_proj

def resampleGeoTiff(GTIFFref,GTIFFslave):
	# Source
	src_filename = GTIFFslave
	src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
	src_proj = src.GetProjection()
	src_geotrans = src.GetGeoTransform()

	# We want a section of source that matches this:
	match_filename = GTIFFref
	match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
	match_proj = match_ds.GetProjection()
	match_geotrans = match_ds.GetGeoTransform()
	width = match_ds.RasterXSize
	height = match_ds.RasterYSize

	dst = gdal.GetDriverByName('MEM').Create('',width, height, 1, gdalconst.GDT_Float32)

	dst.SetGeoTransform( match_geotrans )
	dst.SetProjection( match_proj)

	# Do the work
	gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
	dataslave = dst.ReadAsArray()
	dataref = match_ds.ReadAsArray()
	del dst # Flush
	del src
	return dataref,dataslave,width,height,match_geotrans,match_proj

def ExportGeoTiff(name,data,width,height,match_geotrans,match_proj):
	# Output / destination
	outdriver = gdal.GetDriverByName("GTiff")
	outdata   = outdriver.Create(name, width, height, 1, gdalconst.GDT_Float32)
	outdata.GetRasterBand(1).WriteArray(data)
	outdata.GetRasterBand(1).SetNoDataValue(np.nan)
	outdata.SetGeoTransform(match_geotrans)
	outdata.SetProjection(match_proj)
	return outdata