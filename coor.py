from osgeo import gdal
import tifffile as tiff
import numpy as np


class Coor:
    def __init__(self, coor_path, create_path, nocoor_path):
        self.coor_path = coor_path
        self.create_path = create_path
        self.nocoor_path = nocoor_path

# coor_path = '/gfdc_data/YR/building/result/0_before.tif'
# create_path = '/gfdc_data/YR/building/result/0_before_predict_coor.tif'
# nocoor_path = '/gfdc_data/YR/building/result/0_before_predict.tif'


    def array2raster(self):
        coorRaster = gdal.Open(self.coor_path)
        strRasterFile = self.create_path
        array = np.array(tiff.imread(self.nocoor_path))

        if "int8" in array.dtype.name:
            datatype = gdal.GDT_Byte
        if "int16" in array.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        rows = array.shape[0]
        cols = array.shape[1]
        rows_coor = np.array(tiff.imread(self.coor_path)).shape[0]
        cols_coor = np.array(tiff.imread(self.coor_path)).shape[1]
        driver = gdal.GetDriverByName('Gtiff')
        im_bands = coorRaster.RasterCount
        outRaster = driver.Create(strRasterFile, cols, rows, 1, datatype)
        # outRaster = driver.Create(strRasterFile, cols_coor, rows_coor, 1, datatype)

        # NoCoorRaster = gdal.Open('/data1/yr/python/gdal/test/image/mask02.tif')
        strWkt = coorRaster.GetProjectionRef()
        dGeoTransform = coorRaster.GetGeoTransform()
        # print(strWkt)
        outRaster.SetGeoTransform(dGeoTransform)
        outRaster.SetProjection(strWkt)
        outRaster.GetRasterBand(1).WriteArray(array)
        outRaster.FlushCache()
        print("\nclose dataset\n")


# def test():
#     tmp1 = tiff.imread(coor_path)
#     tmp = np.zeros((tmp1.shape[0], tmp1.shape[1])).astype(np.float32)
#     tmp[:512*(tmp.shape[0]//512), :512*(tmp.shape[1]//512)] = tiff.imread('/gfdc_data/YR/building/predict/prd.tif')
#     tiff.imsave('/gfdc_data/YR/building/predict/tmp.tif', tmp)


# if __name__ == "__main__":
    # test()
    # array2raster()