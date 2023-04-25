#%%!/scratch/users/nifuchs/bin/anaconda2/ipython
# -*- coding: utf-8 -*-
'''
DEM Pond prediction
Terrain and flow accumulation approach, based on pysheds algorithm

created by: Niels Fuchs, niels.fuchsœuni-hamburg.de
edited by: Torbjörn Kagel, torbjoern.kage@studium.uni-hamburg.de
'''
from pysheds.grid import Grid
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import tqdm
from matplotlib import colors
from skimage.feature import peak_local_max
import rasterio as rio
import rioxarray as rxr
import geopandas as gp
from shapely.geometry import mapping
from rasterio.enums import Resampling
import sys
import os
# from import_export_geotiff import ImportGeoTiff, ExportGeoTiff
import pandas as pd
import seaborn as sns
import datetime
import matplotlib.colors as colors
import matplotlib
import datetime
#%%
def pysheds_calculations(directory, first_date, ref_date, mask):
    #for testing:
    # directory = "/home/torka/PraktikumAWI/"
    # mask = "leg4_icefloe_leftCutout.shp"
    # first_date = "20200321"
    # ref_date = "20200630"
    #----------

    os.chdir(directory)
    DEMname = first_date + "_01_DEM_int_PS_crop_0.5m_shift_crop_to_shape.tif"#filledNan_crop_to_shape.tif"
    class_file = ref_date + '_01_main_classes_crop_to_shape.tif'
    grid = Grid.from_raster(DEMname)
    dem = grid.read_raster(DEMname)
    # Conditioning DEM, resolve artifacts/ flat regions
    # ----------------------
    # Fill pits  (regions of cells for which every surrounding cell is at a higher elevation)in DEM
    pit_filled_dem = grid.fill_pits(dem)
    pits = grid.detect_depressions(dem)
    # Fill depressions in DEM
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    print('DEM flooded...')
    # # Resolve flats (regions of cells for which every surrounding cell is at the same elevation or higer) in DEM
    # inflated_dem = grid.resolve_flats(flooded_dem)
    # # Determine D8 flow directions from DEM
    # # ----------------------
    # # Specify directional mapping
    # dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    # # Compute flow directions
    # # -------------------------------------
    # fdir = grid.flowdir(pit_filled_dem, dirmap=dirmap)
    # print('Flow directions calculated...')
    # # Calculate flow accumulation
    # # --------------------------
    # acc = grid.accumulation(fdir, dirmap=dirmap)
    # #############################################


    ### select all areas deeper 5cm
    depth=flooded_dem-dem
    depth[depth == 0] = np.nan
    print('depths determined...')
    depth = np.array(depth)


    ########### terrain approach statistics and plot

    rxr_clas = rxr.open_rasterio(class_file)
    rxr_depths = rxr.open_rasterio(DEMname) 
    floe_shape=gp.read_file(mask)
    clas=rxr_clas.rio.clip(floe_shape.geometry).values[0,:,:]
    clas = np.where(clas == 255, np.nan, np.where(clas == 0, 0, 1))

    raster_tmp = np.invert(np.isnan(rxr_depths))
    raster = rxr_depths.where(raster_tmp,0)
    hydr_tmp=raster.rio.clip(floe_shape.geometry).values[0,:,:]

    clas = np.rint(clas)

    hydr_tmp = np.isnan(hydr_tmp)
    hydr_tmp = np.where(hydr_tmp,np.nan,1)
    depth = np.where(depth>0.03,0,1)
    hydr = np.where(hydr_tmp==1,depth,hydr_tmp)

   
    
    #hydr plot
    # col_dict_hydr={0:"dodgerblue",1:"whitesmoke"}
    # cm_hydr = colors.ListedColormap([col_dict_hydr[x] for x in col_dict_hydr.keys()])
    # labels_hydr = np.array(["meltpond","no meltpond"])
    # len_lab_hydr = len(labels_hydr)
    # norm_bins_hydr = np.sort([*col_dict_hydr.keys()]) + 0.5
    # norm_bins_hydr = np.insert(norm_bins_hydr, 0, np.min(norm_bins_hydr) - 1.0)
    # norm_hydr = matplotlib.colors.BoundaryNorm(norm_bins_hydr, len_lab_hydr, clip=True)
    # fmt_hydr = matplotlib.ticker.FuncFormatter(lambda x, pos: labels_hydr[norm_hydr(x)])
    
    # fig,ax= plt.subplots()
    # plot = ax.imshow(hydr,cmap=cm_hydr,norm=norm_hydr)
    # diff_hydr = norm_bins_hydr[1:] - norm_bins_hydr[:-1]
    # tickz_hydr = norm_bins_hydr[:-1] + diff_hydr / 2
    # # cb_hydr = fig.colorbar(plot, format=fmt_hydr, ticks=tickz_hydr,ax=ax,location='bottom',shrink=0.4)    
    # ax.axis('off')
    # plt.tight_layout()
    # plt.show()

    # #preparing and exporting an accuracy map
    accuracy_map = np.empty((len(clas),len(clas[0])))
    accuracy_map[:] = -1
    
    TP=np.sum(np.logical_and(clas==0, hydr==0))
    FP=np.sum(np.logical_and(clas==1, hydr==0))
    FN=np.sum(np.logical_and(clas==0, hydr==1))
    TN=np.sum(np.logical_and(clas==1, hydr==1))   #subtract all pixels outside of clipped area
        
    precision=(TP/float(TP+FP))*100
    recall=(TP/float(TP+FN))*100
    accuracy=((TP+TN)/float(TP+FP+FN+TN))*100
    bal_accuracy=((TP/float(TP+TN)+TN/float(TN+FP))/2)*100
    fscore= 2*np.array(precision)*np.array(recall)/(np.array(precision)+np.array(recall))
    
    # for i in tqdm.tqdm(range(0,len(clas))):
    #     for j in range(0,len(clas[0])):
    #         if clas[i,j] == 1 and hydr[i,j] == 1:
    #             accuracy_map[i,j] = 0 #TN
    #         elif clas[i,j] == 1 and hydr[i,j] == 0:
    #             accuracy_map[i,j] = 1 #FP
    #         elif clas[i,j] == 0 and hydr[i,j] == 1:
    #             accuracy_map[i,j] = 2 #FN
    #         elif clas[i,j] == 0 and hydr[i,j] == 0:
    #             accuracy_map[i,j] = 3 #TP

    # accuracy_map = np.where(accuracy_map == -1, np.nan, accuracy_map)

    
    # col_dict={0:"green",1:"red",2:"yellow",3:"darkblue"}
    # cm = colors.ListedColormap([col_dict[x] for x in col_dict.keys()])
    # labels = np.array(["TN","FP","FN","TP"])
    # len_lab = len(labels)
    # norm_bins = np.sort([*col_dict.keys()]) + 0.5
    # norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    # print(norm_bins)
    # norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
    # fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
    
    # fig, axs =plt.subplots(1,2,figsize=(10,5), gridspec_kw={'width_ratios': [2, 1]})
    # fig.suptitle("Pysheds confusion \n"+ first_date + " vs. " + ref_date)
    # im = axs[0].imshow(accuracy_map, cmap=cm, norm=norm)
    # axs[0].text(-0.18,1,'a',ha='left',va='center',transform=axs[0].transAxes,fontsize=14,weight="bold")
  
    # diff = norm_bins[1:] - norm_bins[:-1]
    # tickz = norm_bins[:-1] + diff / 2
    # cb = fig.colorbar(im, format=fmt, ticks=tickz,ax=axs[0],location='left',shrink=0.8)

    # axs[0].axis("off")

    # #accuracy_tif = ExportGeoTiff(first_date + 'vs' + ref_date + '_RichDEM_accuracy.tif',accuracy_map,width,height,match_geotrans,match_proj)
    # #accuracy_tif = ExportGeoTiff(first_date + 'vs' + ref_date + '_RichDEM_accuracy.tif',accuracy_map,width,height,match_geotrans,match_proj)

    # data = {'y_Actual':clas.flatten("F"),'y_Predicted':hydr.flatten("F")}
    # df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    # df = df.dropna()
    # df = df[(df["y_Actual"] != 2) & (df["y_Actual"] != 3)]

    # cf_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'], normalize='index')
    # axs[1]= sns.heatmap(cf_matrix, annot=True, cmap='Blues',square=True,cbar=False)
    # axs[1].text(-0.1,1.1,'b',ha='left',va='center',transform=axs[1].transAxes,fontsize=14,weight="bold")
    # #title = 'Confusion Matrix - RichDEM \n ' + first_date + ' vs ' + ref_date
    # #ax.set_title(title);
    # axs[1].set_xlabel('\nPredicted Values')
    # axs[1].set_ylabel('Actual Values ');
    # ## Ticket labels - List must be in alphabetical order
    # axs[1].xaxis.set_ticklabels(['True','False'])
    # axs[1].yaxis.set_ticklabels(['True','False'])
    ## Display the visualization of the Confusion Matrix.
    # plt.show()

    #----------------------------------

    # hydr,width,height,match_geotrans,match_proj = ImportGeoTiff('20200321_01_DEM_int_PS_crop_0.5m_shift_crop_to_shape_richDEM_basin_filled_with_pit_acc.tif') <- for accuracy map with filled pits
    # very inefficient way to calc accuracy map!
    # -------------------------------
    # precision,recall,etc vs depth plot
    # not used anymore, still here for retro reasons
    # fig, ax = plt.subplots(1)
    # fig.set_size_inches(10, 8, forward=True)
    # fig.set_dpi(100)
    # fig.suptitle(DEMname[0:8] + ' vs ' + class_file[0:8] + ' pysheds', fontsize=20)
    # ax.plot(depth_list[:11], np.array(precision[:11])*100, lw=2, color='teal', label='precision')
    # ax.plot(depth_list[10:], np.array(precision[10:])*100, lw=2, ls='--', color='teal', label='precision, DEM pits')
    # ax.plot(depth_list, np.array(recall)*100, lw=2, color='peru', label='recall')
    # ax.plot(depth_list, np.array(accuracy)*100, lw=2, color='gold', label='accuracy')
    # ax.plot(depth_list, np.array(bal_accuracy)*100, lw=2, color='green', label='bal accuracy')
    # ax.plot(depth_list, 2*np.array(precision)*np.array(recall)/(np.array(precision)+np.array(recall))*100., lw=2, color='crimson', label='F-score')
    # ax.legend(fancybox=True, fontsize=20, bbox_to_anchor=(1, 1))
    # ax.set_xlabel('Threshold\nminimum basin depth [m]', fontsize=20)
    # ax.set_ylabel('Pond prediction\nscore [%]', fontsize=20)
    # ax.xaxis.set_tick_params('both', labelsize=18)
    # ax.yaxis.set_tick_params('both', labelsize=18)
    # ax.grid()

    # fig, ax = plt.subplots()
    # ax.scatter(recall, precision, color='purple')
    # #add axis labels to plot
    # ax.set_title('Precision-Recall Curve')
    # ax.set_ylabel('Precision')
    # ax.set_xlabel('Recall')
    # #display plot
    # plt.show()
    
    #-------------------------------
    
    
    return(precision,recall,accuracy,fscore,bal_accuracy)
    
### user inputs
first_date = ['20200321','20200423','20200510'] 
ref_date = ['20200630', '20200707', '20200717', '20200722'] 
directory = "/home/torka/PraktikumAWI/"
mask = "leg4_icefloe_leftCutout.shp"

format_first_date = [datetime.datetime.strptime(i, '%Y%m%d') for i in first_date]
str_first_date = [i.strftime('%d.%m.%Y') for i in format_first_date]
format_ref_date = [datetime.datetime.strptime(i, '%Y%m%d') for i in ref_date]
str_ref_date = [i.strftime('%d.%m.%Y') for i in format_ref_date]

precision_list = []
recall_list = []
accuracy_list = []
fscore_list = []
balacc_list = []
d_optimal_list = []
for ref in ref_date:
    for first in first_date:
        precision,recall,accuracy,fscore,bal_accuracy= pysheds_calculations(directory, first, ref, mask)
        precision_list.append(precision)
        recall_list.append(recall)
        accuracy_list.append(accuracy)
        fscore_list.append(fscore)
        balacc_list.append(bal_accuracy)


#%%

shape = (len(ref_date),len(first_date))
precision_array = np.array(precision_list)
precision_array = precision_array.reshape(shape)
ax = sns.heatmap(precision_array, annot=True, cmap='magma_r',vmin=25,vmax=50,fmt=".1f",annot_kws={"size": 12})
# ax.set_title('Precision Matrix - Pysheds \n with d=0.13' );
ax.set_xlabel('Prediction dates')
ax.set_ylabel('Reference dates');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(str_first_date)
ax.yaxis.set_ticklabels(str_ref_date)
ax.invert_yaxis()
## Display the visualization of the Confusion Matrix.
plt.show()
#%%
shape = (len(ref_date),len(first_date))
recall_array = np.array(recall_list)
recall_array = recall_array.reshape(shape)

ax = sns.heatmap(recall_array, annot=True, cmap='magma_r',vmin=65,vmax=95,fmt=".1f",annot_kws={"size": 12})
# ax.set_title('Recall Matrix - Pysheds \n with d=0.13');
ax.set_xlabel('Prediction dates')
ax.set_ylabel('Reference dates');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(str_first_date)
ax.yaxis.set_ticklabels(str_ref_date)
ax.invert_yaxis()
## Display the visualization of the Confusion Matrix.
plt.show()
#%%
# shape = (len(ref_date),len(first_date))

# accuracy_array = np.array(accuracy_list)
# accuracy_array = accuracy_array.reshape(shape)

# ax = sns.heatmap(accuracy_array, annot=True, cmap='magma_r',robust=True,fmt=".1f",annot_kws={"size": 12})
# ax.set_title('Accuracy Matrix - Pysheds \n with d=0.13');
# ax.set_xlabel('Prediction dates')
# ax.set_ylabel('Reference dates');

# ## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(str_first_date)
# ax.yaxis.set_ticklabels(str_ref_date)
# ax.invert_yaxis()
## Display the visualization of the Confusion Matrix.
# plt.show()
#%%
shape = (len(ref_date),len(first_date))

fscore_array = np.array(fscore_list)
fscore_array = fscore_array.reshape(shape)

ax = sns.heatmap(fscore_array, annot=True, cmap='magma_r',vmax=60,vmin=42,fmt=".1f",annot_kws={"size": 12})
# ax.set_title('F-score Matrix - Pysheds \n with d=0.13');
ax.set_xlabel('Prediction dates')
ax.set_ylabel('Reference dates');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(str_first_date)
ax.yaxis.set_ticklabels(str_ref_date)
ax.invert_yaxis()
## Display the visualization of the Confusion Matrix.
plt.show()
#%%

# shape = (len(ref_date),len(first_date))
# balacc_array = np.array(balacc_list)
# balacc_array = balacc_array.reshape(shape)

# ax = sns.heatmap(balacc_array, annot=True, cmap='magma_r',robust=True,fmt=".1f",annot_kws={"size": 12})
# ax.set_title('Balanced Accuracy Matrix - Pysheds \n with d=0.13');
# ax.set_xlabel('Prediction dates')
# ax.set_ylabel('Reference dates');

# ## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(first_date)
# ax.yaxis.set_ticklabels(ref_date)
# ax.invert_yaxis()
# ## Display the visualization of the Confusion Matrix.
# plt.show()
#%%


######### accumulation approach statistics and plot

# precision=[]
# recall=[]
# accuracy=[]
# depth_list=[]

# rxr_map = rxr.open_rasterio(DEMname)  # use DEM file for georeference metadata, fill with pond data later
# rxr_clas = rxr.open_rasterio(class_file)

# # same as before
# rxr_map.values[0,:,:]=pond_vol_bool
# rxr_map_reproj = rxr_map.rio.reproject_match(rxr_clas,Resampling.nearest)
# clas=rxr_clas.rio.clip(floe_shape.geometry).values[0,:,:]
# clas=clas==0
# clas_back = rxr_clas.rio.clip(floe_shape.geometry).values[0,:,:]
# rxr_map=rxr_map_reproj.rio.clip(floe_shape.geometry).values[0,:,:]

# hydr=rxr_flooded>0.05   # let's only consider areas deeper than 5cm
# TP_half=np.sum(np.logical_and(clas, rxr_map==1))    # not fully filled basins
# TP_full=np.sum(np.logical_and(clas, rxr_map==2))    # overflowing basins
# TP_both=np.sum(np.logical_and(clas, np.isin(rxr_map,[1,2])))
# TP_05=np.sum(np.logical_and(clas, hydr))    # 5cm threshold
# FP_05=np.sum(np.logical_and(hydr, clas==False))
# FP_half=np.sum(np.logical_and(clas==False, rxr_map==1))
# FP_full=np.sum(np.logical_and(clas==False, rxr_map==2))
# FP_both=np.sum(np.logical_and(clas==False, np.isin(rxr_map,[1,2])))


# precision_half=TP_half/float(TP_half+FP_half)
# precision_full=TP_full/float(TP_full+FP_full)
# precision_both=TP_both/float(TP_both+FP_both)
# precision_05=TP_05/float(TP_05+FP_05)

# # plot accumulation volume map

# fig, ax = plt.subplots(figsize=(8,6))
# fig.patch.set_alpha(0)
# plt.grid('on', zorder=0)
# im = ax.imshow(acc*volume, extent=grid.extent, zorder=2,
#                cmap='cubehelix',
#                norm=colors.LogNorm(volume, (acc*volume).max()),
#                interpolation='bilinear')
# cbar=plt.colorbar(im, ax=ax)
# plt.title('Flow Accumulation', size=14)
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.tight_layout()
# cbar.ax.tick_params(labelsize=16)
# cbar.set_label(label=r'Upstream volume [m$^3$]', size=18)
# plt.show()


# depth_bool=depth>0.25

    # depth_bool=np.int64(depth_bool)

    # labels=ndi.label(depth_bool)    #gives every area(!) deeper than 5cm a unique label
 
    # uniq_lab, counts = np.unique(labels[0], return_counts=True)   #returns pixel counts (equals size)

    # pond_label = np.ones(labels[0].shape)*-32767    # setting -32767 as NaN

    # # remove all labeled area smaller 100 pixels, re-label
    # for n, l in tqdm.tqdm(enumerate(uniq_lab[counts>100])):
    #     pond_label[labels[0]==l]=n
        
    # vol = np.zeros(np.sum(counts>100))
    # ac = np.zeros(np.sum(counts>100))
    # overflow = np.zeros(np.sum(counts>100))

    # pond_label_new = pond_label.copy()
    # pond_vol_bool = np.ones(pond_label_new.shape)*-32767



    # # area and volume per pixel, assumes rectangular, cartesian grid
    # area = grid.affine[0]*grid.affine[4]*-1
    # volume = grid.affine[0]*grid.affine[4]*-1*grid.affine[0]

    # # volume and accumulation volume within ponds. Assumes that meltwater column height equals GSD
    # for i in tqdm.tqdm(range(np.sum(counts>100))):
    #     vol[i] = np.sum(depth[pond_label==i])*area #summed pixel-depths * pixel area
    #     ac[i] = np.sum(max_acc[pond_label==i])*volume

    # overflow=ac/(vol+ac)


    # ### recalculate theoertical pond expansion based on available melt water. Returns 1 for filled pond areas, 2 for overflowing pond areas and -32767 where not enough melt water is available
    # for i in tqdm.tqdm(range(np.sum(counts>100))):
    #     if vol[i]>ac[i]:
    #         level=np.max(depth[pond_label==i])
    #         v=0
    #         while v < ac[i]:
    #             level-=grid.affine[0]
    #             v=np.sum(np.clip(depth[pond_label==i]-level,0,None))*area
    #         pond_label_new[np.logical_and(pond_label==i, depth<level)]=-32767
    #         pond_vol_bool[np.logical_and(pond_label==i, depth>=level)]=1
    #     else:
    #         pond_vol_bool[pond_label==i]=2
            
    # pond_vol_bool[pond_label==0]=0
    # print('Potential meltponds classified...')


    # # write files
    # with rio.open(DEMname,'r') as src:
    #     transform = src.transform
    #     meta = src.meta

    # with rio.open(DEMname.rsplit('.',1)[0]+'_pysheds'+'_basin_filled_with_pit_acc.tif','w+',**meta) as dst:
    #      dst.write(pond_label_new==0,1)
         
    # with rio.open(DEMname.rsplit('.',1)[0]+'_pysheds'+'_pit_acc_volume_bool.tif','w+',**meta) as dst:
    #      dst.write(pond_vol_bool,1)
         
    # open files for plotting