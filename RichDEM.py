#%%
import richdem as rd
# from import_export_geotiff import ImportGeoTiff, ExportGeoTiff
import numpy as np
from scipy import ndimage as ndi
import tqdm
# from skimage.feature import peak_local_max
import rasterio as rio
import os
import rioxarray as rxr
import matplotlib.pyplot as plt
from matplotlib import colors

from rasterio.enums import Resampling
import geopandas as gp
#from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import datetime
import matplotlib
#%%
def richdem_calculations(directory, first_date, ref_date, mask):
    #for testing:
    # directory = "/home/torka/PraktikumAWI/"
    # mask = "leg4_icefloe_leftCutout.shp"
    # first_date = "20200423"
    # ref_date = "20200630"
    # ----------
    os.chdir(directory)
    DEMname = first_date
    class_file = ref_date + '_01_main_classes_crop_to_shape.tif'
    # class_file = '20200630_classes_onlyLargePonds.tif'
    dem   = rd.LoadGDAL(DEMname)
    dem_arr = rd.rdarray(dem, no_data=-32767)

    #Fill depressions in the DEM. The data is modified in-place to avoid making
    #an unnecessary copy. This saves both time and RAM. Note that the function
    #has no return value when `in_place=True`.
    accum_d8 = rd.FlowAccumulation(dem_arr, method='D8')
    #d8_fig = rd.rdShow(accum_d8, figsize=(8,5.5), axes=False, cmap='jet')
    flooded_dem = rd.FillDepressions(dem_arr, in_place=False, epsilon=False)

    depth    =  flooded_dem - dem_arr
    depth[depth == 0] = np.nan
    depth = np.array(depth)
    print('depths determined...')

    # plt.imshow(depth)
    # plt.colorbar()
    # plt.show()
    # open files for plotting
    rxr_clas = rxr.open_rasterio(class_file)
    rxr_depths = rxr.open_rasterio(DEMname)
    # rxr_depths = flooded_dem-rxr_depths.values[0,:,:]
    floe_shape=gp.read_file(mask)
    clas=rxr_clas.rio.clip(floe_shape.geometry).values[0,:,:]

    raster_tmp = np.invert(np.isnan(rxr_depths))
    raster = rxr_depths.where(raster_tmp,0)
    hydr_tmp=raster.rio.clip(floe_shape.geometry).values[0,:,:]
    clas = np.where(clas == 255, np.nan, np.where(clas == 0, 0, 1))
    clas = np.rint(clas)

    hydr_tmp = np.isnan(hydr_tmp)
    hydr_tmp = np.where(hydr_tmp,np.nan,1)
    depth = np.where(depth>0.03,0,1)
    hydr = np.where(hydr_tmp==1,depth,hydr_tmp)

  

    #control plot for hydr/clas
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
    #-------
    
    #preparing and exporting an accuracy map
    accuracy_map = np.empty((len(clas),len(clas[0])))
    accuracy_map[:] = -1
    
    TP=np.sum(np.logical_and(clas==0, hydr==0))
    FP=np.sum(np.logical_and(clas==1, hydr==0))
    FN=np.sum(np.logical_and(clas==0, hydr==1))
    TN=np.sum(np.logical_and(clas==1, hydr==1))  #subtract all pixels outside of clipped area
        
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
    
    # col_dict={0:"mediumseagreen",1:"indianred",2:"gold",3:"dodgerblue"}
    # cm = colors.ListedColormap([col_dict[x] for x in col_dict.keys()])
    # labels = np.array(["TN","FP","FN","TP"])
    # len_lab = len(labels)
    # norm_bins = np.sort([*col_dict.keys()]) + 0.5
    # norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    # print(norm_bins)
    # norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
    # fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
    
    # fig, axs =plt.subplots(1,2,figsize=(10,5), gridspec_kw={'width_ratios': [2, 1]})
    # # fig.suptitle("RichDEM confusion \n"+ first_date + " vs. " + ref_date)
    # im = axs[0].imshow(accuracy_map, cmap=cm, norm=norm)   
    # diff = norm_bins[1:] - norm_bins[:-1]
    # tickz = norm_bins[:-1] + diff / 2
    # cb = fig.colorbar(im, format=fmt, ticks=tickz,ax=axs[0],location='left',shrink=0.8)
    # axs[0].text(-0.18,1,'a',ha='left',va='center',transform=axs[0].transAxes,fontsize=14,weight="bold")

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
    # ## Display the visualization of the Confusion Matrix.
    # plt.show()

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
    # -------------------------------
    return(precision,recall,accuracy,fscore,bal_accuracy)

#Load DEM
first_date = ['20200107_01_ALS_DEM_05m_leg4CO_shift_filledNan_crop_to_shape.tif','20200116_01_ALS_DEM_05m_leg4CO_shift_filledNan_crop_to_shape.tif','20200321_ALS_DEM_filledNan_crop_to_shape.tif','20200321_01_DEM_int_PS_crop_05m_shift_crop_to_shape_filledNan_crop_to_shape.tif','20200423_01_ALS_DEM_05m_leg4CO_shift_crop_to_shape.tif','20200423_01_DEM_int_PS_crop_05m_shift_crop_to_shape_filledNan_crop_to_shape.tif'] #20200321, 20200423, 20200510 are valid options
ref_date = ['20200630','20200707','20200717','20200722'] #20200630, 20200704, 20200707, 20200717, 20200722 are valid options
directory = "/home/torka/PraktikumAWI/"
mask = "leg4_icefloe_leftCutout.shp"

# format_first_date = [datetime.datetime.strptime(i, '%Y%m%d') for i in first_date]
# str_first_date = [i.strftime('%d.%m.%Y') for i in format_first_date]
str_first_date = ['07.01\nALS','16.01\nALS','21.03\nALS','21.03\nphoto','23.04\nALS','23.04\nphoto']

format_ref_date = [datetime.datetime.strptime(i, '%Y%m%d') for i in ref_date]
str_ref_date = [i.strftime('%d.%m.%Y') for i in format_ref_date]

precision_list = []
recall_list = []
accuracy_list = []
fscore_list = []
d_optimal_list = []
balacc_list = []
for ref in ref_date:
    for first in first_date:
        precision,recall,accuracy,fscore,bal_accuracy= richdem_calculations(directory, first, ref, mask)
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
# ax.set_title('Precision Matrix - RichDEM \n with d=0.13' );
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
# ax.set_title('Recall Matrix - RichDEM \n with d=0.13');
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
# ax.set_title('Accuracy Matrix - RichDEM \n with d=0.13');
# ax.set_xlabel('Prediction dates')
# ax.set_ylabel('Reference dates');

# ## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(str_first_date)
# ax.yaxis.set_ticklabels(str_ref_date)
# ax.invert_yaxis()
# ## Display the visualization of the Confusion Matrix.
# plt.show()
#%%
shape = (len(ref_date),len(first_date))

fscore_array = np.array(fscore_list)
fscore_array = fscore_array.reshape(shape)

ax = sns.heatmap(fscore_array, annot=True, cmap='magma_r',vmax=60,vmin=42,fmt=".1f",annot_kws={"size": 12})
# ax.set_title('F-score Matrix - RichDEM \n with d=0.13');
ax.set_xlabel('Prediction dates')
ax.set_ylabel('Reference dates');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(str_first_date)
ax.yaxis.set_ticklabels(str_ref_date)
ax.invert_yaxis()
## Display the visualization of the Confusion Matrix.
plt.show()
# plt.savefig("fscore_RichDEM.png")
#%%
# shape = (len(ref_date),len(first_date))
# balacc_array = np.array(balacc_list)
# balacc_array = balacc_array.reshape(shape)

# ax = sns.heatmap(balacc_array, annot=True, cmap='magma_r',robust=True,fmt=".1f")
# ax.set_title('Balanced Accuracy Matrix - RichDEM \n with d=0.13');
# ax.set_xlabel('Prediction dates')
# ax.set_ylabel('Reference dates');

# ## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(first_date)
# ax.yaxis.set_ticklabels(ref_date)
# ax.invert_yaxis()
# ## Display the visualization of the Confusion Matrix.
# plt.show()
#%%
