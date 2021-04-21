#!/usr/bin/env python
# coding: utf-8

import shutil
import re
import os
import numpy as np
import functools
import pandas as pd
import SimpleITK as sitk
from skimage.measure import label 
from collections import Counter

FILE_NUM = 3

## calculate mean and median from all images
def calculateStats(directory):
    
    voxelSum = 0.0
    voxelSumSq = 0.0
    numVoxels = 0
    
    maxVal = float('-inf')
    maxFile = None
    minVal = float('inf')
    minFile = None
    
    for subdir in os.listdir(os.fsencode(directory)):
        subdirname = os.fsdecode(subdir)
        if not subdirname.startswith("."):
            full_subdir_path = os.path.join(directory, subdirname)
            for file in os.listdir(os.fsencode(full_subdir_path)):
                filename = os.fsdecode(file)
                if filename.endswith(".nii"):
                    if filename.startswith("volume"): 
                        full_file_path = os.path.join(full_subdir_path, filename)
                        img = sitk.GetArrayFromImage(sitk.ReadImage(full_file_path))
                        voxelSum += np.sum(img)
                        voxelSumSq += np.sum(np.square(img))
                        numVoxels += img.shape[0] * img.shape[1] * img.shape[2]
                        ma = np.max(img)
                        if ma > maxVal:
                            maxVal = ma
                            maxFile = full_file_path
                        mi = np.min(img)
                        if mi < minVal:
                            minVal = mi  
                            minFile = full_file_path
    
    mean = voxelSum / numVoxels
    stddev = (voxelSumSq / numVoxels - mean**2)**(0.5)
                    
    return mean, stddev, minVal, maxVal, minFile, maxFile

def printStats(img):
    print("Shape:", img.shape)
    print("Min:", np.min(img))
    print("Max:", np.max(img))
    print("Mean:", np.mean(img))
    print("StdDev:", np.std(img))
    print("Min index:", np.unravel_index(np.argmin(img), img.shape))
    print("Max index:", np.unravel_index(np.argmax(img), img.shape))


def rebuildNii(directory, folder_name, mean, stddev):
    img = None
    final_seg = None
    segs = []

    for file in os.listdir(os.fsencode(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".nii"):
            if filename.startswith("volume"): 
                img_itk = sitk.ReadImage(os.path.join(directory, filename))
                img = sitk.GetArrayFromImage(img_itk)
                print('img loaded from:', filename, img.shape) #(512, 512, 150)
            else:
                seg_itk = sitk.ReadImage(os.path.join(directory, filename))
                seg = sitk.GetArrayFromImage(seg_itk)
                print('seg loaded from:', filename, seg.shape)             
                segs.append(seg)
                
    if(len(segs) == 0):
        assert "seg cannot empty"
    elif(len(segs) == 1):
        mid_seg = segs[0]
    else:
        mid_seg = functools.reduce(lambda a, b: np.bitwise_or(a, b), segs)
            
        
    #normalize image
    img = (img - mean) / stddev
    print(img.shape)
    final_img = sitk.GetImageFromArray(img)
    final_img.CopyInformation(img_itk)
    final_seg = sitk.GetImageFromArray(mid_seg)
    final_seg.CopyInformation(seg_itk)

#     print(final_img.GetSize())

    if os.path.exists(folder_name):
        print('folder already exists:', folder_name)
#         sitk.WriteImage(final_img, os.path.join(folder_name,"img.nii"))
#         print('img saved',os.path.join(folder_name, "img.nii"))
#         sitk.WriteImage(final_seg, os.path.join(folder_name, "seg.nii"))
#         print('seg saved',os.path.join(folder_name, "seg.nii"))
    else:
        os.makedirs(folder_name)
        sitk.WriteImage(final_img, os.path.join(folder_name,"img.nii"))
        print('img saved',os.path.join(folder_name, "img.nii"))
        sitk.WriteImage(final_seg, os.path.join(folder_name, "seg.nii"))
        print('seg saved',os.path.join(folder_name, "seg.nii"))

def rebuildNii_compress(directory, folder_name, mean, stddev):
    img = None
    final_seg = None
    segs = []

    for file in os.listdir(os.fsencode(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".nii"):
            if filename.startswith("volume"): 
                img_itk = sitk.ReadImage(os.path.join(directory, filename))
                img = sitk.GetArrayFromImage(img_itk)
                print('img loaded from:', filename, img.shape) #(512, 512, 150)
            else:
                seg_itk = sitk.ReadImage(os.path.join(directory, filename))
                seg = sitk.GetArrayFromImage(seg_itk)
                print('seg loaded from:', filename, seg.shape)        
                segs.append(seg)
                
    if(len(segs) == 0):
        assert "seg is empty"
#         final_seg = np.zeros(img.shape)
    elif(len(segs) == 1):
        mid_seg = segs[0]
    else:
        mid_seg = functools.reduce(lambda a, b: np.bitwise_or(a, b), segs)
            
    #D, H, W = img.shape
#     H, W, D = img.shape
    
    #hack to move depth to 1st dim
#     if D == H:
#         img = img.transpose(2, 0, 1)
#         mid_seg = mid_seg.transpose(2, 0, 1)
#         D, W = W, D
        
    #normalize image
    img = (img - mean) / stddev
#     print(img.shape)
    final_img = sitk.GetImageFromArray(img)
    final_img.CopyInformation(img_itk)
    final_seg = sitk.GetImageFromArray(mid_seg)
    final_seg.CopyInformation(seg_itk)

#     print(final_img.GetSize())

    if os.path.exists(folder_name):
        print('folder already exists:', folder_name)
#         sitk.WriteImage(final_img, os.path.join(folder_name,"img.nii"))
#         print('img saved',os.path.join(folder_name, "img.nii"))
#         sitk.WriteImage(final_seg, os.path.join(folder_name, "seg.nii"))
#         print('seg saved',os.path.join(folder_name, "seg.nii"))
    else:
        os.makedirs(folder_name)
        sitk.WriteImage(final_img, os.path.join(folder_name,"img.nii.gz"))
        print('img saved',os.path.join(folder_name, "img.nii.gz"))
        sitk.WriteImage(final_seg, os.path.join(folder_name, "seg.nii.gz"))
        print('seg saved',os.path.join(folder_name, "seg.nii.gz"))

# add ID to filename
def filename_add_ID(source):
    """
    MUST RUN rebuildNii first.
    add ID to img.nii and seg.nii , only rename if ID is numeric, can only be used once
    source: can be any source
    
    """
    for root, dirs, files in os.walk(source, topdown=False):
        for file in files:
            if 'DS' not in file:
                if 'gz' not in file:
                    dir_name = os.path.basename(root)
                    if not dir_name.isnumeric():
                        print('only numeric IDs can be added but',dir_name, "was found")
                        return
                    else:
                        file_name = os.path.splitext(file)[0]
                        assert 'img_' not in file_type, 'this function can only be used once'
                        extension = os.path.splitext(file)[1]
                        os.rename(root+"/"+file,root+"/"+ file_name +'_'+ dir_name + extension)
                else:
                    dir_name = os.path.basename(root)
                    if not dir_name.isnumeric():
                        print('only numeric IDs can be added but',dir_name, "was found")
                        return
                    else:
                        file_type = os.path.splitext(file)[0].split('.')[0]
                        assert 'img_' not in file_type, "this function can only be used once"
                        extension_0 = os.path.splitext(file)[0].split('.')[1]
                        extension_1 = os.path.splitext(file)[1]
                        os.rename(root + "/" + file,root+"/"+ file_type +'_'+ dir_name + '.' + extension_0 + extension_1)      
                        

## bring all files to a new directory, filaname needs to be unique
def move_up_to_parentDir(source, newPath):
    """
    NEED TO RUN rebuildNii_compress and filename_add_ID first 
    bring files from subdir to a newPath and remove the original directory
    (subdirs can only be ID_num)
    source: must be the parent dir that contains patient dir with ID_num
    newPath : i.e 'all_sag'
    
    """
    list_subfolders = [f.path for f in os.scandir(source) if f.is_dir()]
    if not any(i.split('/')[-1].isnumeric() for i in list_subfolders):
        print('No patient folder was found')
        return   
        
    if not os.path.exists(os.path.join(source,newPath)):
         os.makedirs(os.path.join(source,newPath))
    for subdir in os.listdir(source):
        if '.DS_' not in subdir and subdir.isnumeric():
            for filename in os.listdir(os.path.join(source,subdir)):
                if '.DS_' not in filename:
                    assert filename != 'img.nii.gz' and filename != 'seg.nii.gz', 'Did you run filename_add_ID first?'
                    src = os.path.join(source,subdir,filename)
                    dst = os.path.join(source,newPath,filename)
                    shutil.move(src, dst)
            shutil.rmtree(os.path.join(source,subdir)) 
    print('move and remove finished')
              

## separate img and seg to imagesTr and labelsTr
def seperate_files(source):
    """
    can only be used after move_up_to_parentDir
    source: must be in a parent directory that has seg and img files
    
    """
    
    assert os.path.exists(os.path.join(source)),'source diretory does not exsist'
    if not os.path.exists(os.path.join(source,'imagesTr')):
         os.makedirs(os.path.join(source, 'imagesTr'))
    if not os.path.exists(os.path.join(source,'labelsTr')):
         os.makedirs(os.path.join(source, 'labelsTr'))
    for filename in os.listdir(source):
        if 'img' in filename:
            shutil.move(os.path.join(source,filename), os.path.join(source,'imagesTr',filename))
        if 'seg' in filename:
            shutil.move(os.path.join(source,filename), os.path.join(source,'labelsTr',filename))    


## build a map from ID and rename for nnunet
def ID_map_imagesTr(source, pj_name, mod):
    """
    CAN ONLY RUN ONCE
    source: must be from imagesTr, imagesTr are processed after move_up_to_parentDir and  
    pj_name: project name
    mod: modality
    
    """
    assert 'imagesTr' in source, 'source must be from imagesTr'
    
    ID_map = {}
    i = 0
    for root, dirs, files in os.walk(source):   
        for file in files:
            if 'DS' not in file:
                value = '{0:03d}'.format(i) + '_' + mod
                file_type = os.path.splitext(file)[0].split('_')[0]
                ID_extract = os.path.splitext(file)[0].split('_')[1]
                ID_num = re.findall('\d+', ID_extract)[0]
                assert '000' not in ID_num, 'wrong ID_number, already converted, can only run once'
#                 if '000' in ID_num:
#                     return
# #                 key = file_type +'_'+ ID_num
#                 else: 
                dst = os.path.join(source, pj_name + '_' + value + '.nii.gz')
                src = os.path.join(source, file)
                ID_map[ID_num] = value
                os.rename(src, dst)
                i += 1
        print('files renamed to', pj_name + '_XXX_'+ mod, 'successfully')
        return ID_map


def dict_to_df_save(data, title):
    assert type(data) is dict, 'source data must be a dictionary'
    df = pd.DataFrame(data.items(), columns=['ID', 'index_map'])
    df.to_csv('./'+ title + '.csv',index=False)
    return df
   

## build a map from ID and rename for nnunet
def ID_map_labelsTr(source, pj_name):
    """
    CAN ONLY RUN ONCE
    source: must be from labelsTr
    pj_name: project name
    mod: modality
    
    """
    assert 'labelsTr' in source, 'source must be from labelsTr'
    
    ID_map = {}
    i = 0
    for root, dirs, files in os.walk(source):   
        for file in files:
            if 'DS' not in file:
                value = '{0:03d}'.format(i)
                file_type = os.path.splitext(file)[0].split('_')[0]
                ID_extract = os.path.splitext(file)[0].split('_')[1]
                ID_num = re.findall('\d+', ID_extract)[0]
                assert '000' not in ID_num, 'wrong ID_number, already converted, cannot proceed'
#                 key = file_type +'_'+ ID_num
                dst = os.path.join(source, pj_name + '_' + value + '.nii.gz')
                src = os.path.join(source, file)
                ID_map[ID_num] = value
                os.rename(src, dst)
                i += 1
        print('files renamed to', pj_name + '_XXX', 'successfully')
        return ID_map


def maxArea(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1 ## +1 is becuase indexfrom[1:] 
    largestCC = largestCC.astype(int)
    return largestCC 


def rebuildNii_max(directory, folder_name, mean, stddev):
    img = None
    final_seg = None
    segs = []

    for file in os.listdir(os.fsencode(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".nii"):
            if filename.startswith("volume"): 
                img_itk = sitk.ReadImage(os.path.join(directory, filename))
                img = sitk.GetArrayFromImage(img_itk)
                print('img loaded from:', filename, img.shape) #(512, 512, 150)
            else:
                seg_itk = sitk.ReadImage(os.path.join(directory, filename))
                seg = sitk.GetArrayFromImage(seg_itk)
                print('seg loaded from:', filename, seg.shape)
#                 seg = seg[:,:,:,0]                
                segs.append(seg)
                
    if(len(segs) == 0):
        assert "seg cannot be empty"
#         final_seg = np.zeros(img.shape)
    elif(len(segs) == 1):
        mid_seg = segs[0]
    else:
        mid_seg = functools.reduce(lambda a, b: np.bitwise_or(a, b), segs)
            
    #normalize image
    img = (img - mean) / stddev
    print(img.shape)
    final_img = sitk.GetImageFromArray(img)
    final_img.CopyInformation(img_itk)
    final_seg = sitk.GetImageFromArray(mid_seg)
    final_seg.CopyInformation(seg_itk)

#     print(final_img.GetSize())

    if os.path.exists(folder_name):
        print('folder already exists:', folder_name)
#         sitk.WriteImage(final_img, os.path.join(folder_name,"img.nii"))
#         print('img saved',os.path.join(folder_name, "img.nii"))
#         sitk.WriteImage(final_seg, os.path.join(folder_name, "seg.nii"))
#         print('seg saved',os.path.join(folder_name, "seg.nii"))
    else:
        os.makedirs(folder_name)
        sitk.WriteImage(final_img, os.path.join(folder_name,"img.nii"))
        print('img saved',os.path.join(folder_name, "img.nii"))
        sitk.WriteImage(final_seg, os.path.join(folder_name, "seg.nii"))
        print('seg saved',os.path.join(folder_name, "seg.nii"))



## find multiple tumors 
def multi_finder(source):
    """
    return a dataframe of patients who has muiltiple tumors
    source: can be any folder

    """
    multi = []
    for path, dirs, files in os.walk(source):
        if len(files) > FILE_NUM:
            for file in files:
                if '.DS' not in file: 
                    multi_ID = re.findall('\d+', file)
                    if len(multi_ID) != 0 and len(multi_ID) >1:
                        ID = int(multi_ID[0])
# #                     print(ID)        
                        multi.append(ID)
#     print(multi)
    # return multi
    dict_multi  = Counter(multi)
    df_multi = pd.DataFrame(dict_multi.items(), columns=['ID', 'Tumor_num'])
    df_multi.to_csv('./multi_tumor_ID.csv',index=False)
    return df_multi

def nii_to_gz(filename):
    """
    nii file has to be read first
    
    """
    f = sitk.ReadImage(filename)
    sitk.WriteImage(f, os.path.splitext(filename)[0] + ".nii.gz")
    os.remove(filename)





