from __future__ import absolute_import, print_function
import os
import sys
import numpy as np
from scipy import ndimage
sys.path.append('./')
import nibabel
import random
import SimpleITK as sitk
import pdb
def border_map(binary_img,neigh):
    """
    Creates the border for a 3D image
    """
    binary_map = np.asarray(binary_img, dtype=np.uint8)
    neigh = neigh
    west = ndimage.shift(binary_map, [-1, 0,0], order=0)
    east = ndimage.shift(binary_map, [1, 0,0], order=0)
    north = ndimage.shift(binary_map, [0, 1,0], order=0)
    south = ndimage.shift(binary_map, [0, -1,0], order=0)
    top = ndimage.shift(binary_map, [0, 0, 1], order=0)
    bottom = ndimage.shift(binary_map, [0, 0, -1], order=0)
    cumulative = west + east + north + south + top + bottom
    border = ((cumulative < 6) * binary_map) == 1
    return border


def border_distance(ref,seg):
    """
    This functions determines the map of distance from the borders of the
    segmentation and the reference and the border maps themselves
    """
    neigh=8
    border_ref = border_map(ref,neigh)
    border_seg = border_map(seg,neigh)
    oppose_ref = 1 - ref
    oppose_seg = 1 - seg
    # euclidean distance transform
    distance_ref = ndimage.distance_transform_edt(oppose_ref)
    distance_seg = ndimage.distance_transform_edt(oppose_seg)
    distance_border_seg = border_ref * distance_seg
    distance_border_ref = border_seg * distance_ref
    return distance_border_ref, distance_border_seg#, border_ref, border_seg

def binary_dice3d(s,g):
    """
    dice score of 3d binary volumes
    inputs: 
        s: segmentation volume
        g: ground truth volume
    outputs:
        dice: the dice score
    """
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert(Ds==Dg and Hs==Hg and Ws==Wg)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0*s0 + 1e-10)/(s1 + s2 + 1e-10)
    return dice

def binary_sens3d (s, g): 
    #computs false negative rate
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert(Ds==Dg and Hs==Hg and Ws==Wg)
    num=np.sum(np.multiply(g, s ))
    denom=np.sum(g)
    if denom==0:
        return 1
    else:
        return  num*1.0/denom

def binary_spec3d (s,g): 
    #computes false positive rate
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert(Ds==Dg and Hs==Hg and Ws==Wg)
    num=np.sum(np.multiply(g==0, s ==0))
    denom=np.sum(g==0)
    if denom==0:
        return 1
    else:
        return  num*1.0/denom

def binary_haus3d(s,g):
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert(Ds==Dg and Hs==Hg and Ws==Wg)
    ref_border_dist, seg_border_dist = border_distance(g,s)
    border_ref, border_seg = border_map(g,8), border_map(s,8) 
    seg_values = ref_border_dist[border_seg>0]
    ref_values = seg_border_dist[border_ref>0]
    if seg_values.size==0 or ref_values.size == 0:
	return np.nan
    else:
        hausdorff95_distance = np.max([np.percentile(seg_values,95),
                                           np.percentile(ref_values,95)])

    return hausdorff95_distance


def load_3d_volume_as_array(filename, with_header = False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
        with_header: return affine and hearder infomation
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    if(with_header):
        return data, img.affine, img.header
    else:
        return data

def get_ground_truth_names(g_folder, patient_names_file, year = 17):
    assert(year == 17)
    with open(patient_names_file) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content]
    full_gt_names = []
    for patient_name in patient_names:
        patient_dir = os.path.join(g_folder, patient_name)
        img_names   = os.listdir(patient_dir)
        gt_name = None
        for img_name in img_names:
                if 'seg.' in img_name:
                    gt_name = img_name
                    break
        gt_name = os.path.join(patient_dir, gt_name)
        full_gt_names.append(gt_name)
    return full_gt_names

def get_segmentation_names(seg_folder, patient_names_file):
    with open(patient_names_file) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content]
    full_seg_names = []
    for patient_name in patient_names:
        seg_name = os.path.join(seg_folder, patient_name + '.nii.gz')
        full_seg_names.append(seg_name)
    return full_seg_names




def dice_of_brats_data_set(gt_names, seg_names, type_idx):
    assert(len(gt_names) == len(seg_names))
    dice_all_data = []
    for i in range(len(gt_names)):
        g_volume = load_3d_volume_as_array(gt_names[i])
        s_volume = load_3d_volume_as_array(seg_names[i])
        dice_one_volume = []
        if(type_idx ==0): # enhancing tumor
            s_volume[s_volume == 2] = 0
            g_volume[g_volume == 2] = 0
            s_volume[s_volume == 1] = 0
            g_volume[g_volume == 1] = 0
            temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
            dice_one_volume = [temp_dice]
        elif(type_idx ==1): # whole tumor
            temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
            dice_one_volume = [temp_dice]
        elif(type_idx == 2): # tumor core
            s_volume[s_volume == 2] = 0
            g_volume[g_volume == 2] = 0
            temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
            dice_one_volume = [temp_dice]
        else:
            for label in [1, 2, 3, 4]: # dice of each class
                temp_dice = binary_dice3d(s_volume == label, g_volume == label)
                dice_one_volume.append(temp_dice)
        dice_all_data.append(dice_one_volume)
    return dice_all_data
    
def sens_of_brats_data_set(gt_names, seg_names, type_idx):
    assert(len(gt_names) == len(seg_names))
    sens_all_data = []
    for i in range(len(gt_names)):
        g_volume = load_3d_volume_as_array(gt_names[i])
        s_volume = load_3d_volume_as_array(seg_names[i])
        sens_one_volume = []
        if(type_idx ==0): # enhancing tumor
            s_volume[s_volume == 2] = 0
            g_volume[g_volume == 2] = 0
            s_volume[s_volume == 1] = 0
            g_volume[g_volume == 1] = 0
            temp_sens = binary_sens3d(s_volume > 0, g_volume > 0)
            sens_one_volume = [temp_sens]
        elif(type_idx ==1): # whole tumor
            temp_sens = binary_sens3d(s_volume > 0, g_volume > 0)
            sens_one_volume = [temp_sens]
        elif(type_idx == 2): # tumor core
            s_volume[s_volume == 2] = 0
            g_volume[g_volume == 2] = 0
            temp_sens = binary_sens3d(s_volume > 0, g_volume > 0)
            sens_one_volume = [temp_sens]
        else:
            for label in [1, 2, 3, 4]: # sens of each class
                temp_sens = binary_sens3d(s_volume == label, g_volume == label)
                sens_one_volume.append(temp_sens)
        sens_all_data.append(sens_one_volume)
    return sens_all_data

def spec_of_brats_data_set(gt_names, seg_names, type_idx):
    assert(len(gt_names) == len(seg_names))
    spec_all_data = []
    for i in range(len(gt_names)):
        g_volume = load_3d_volume_as_array(gt_names[i])
        s_volume = load_3d_volume_as_array(seg_names[i])
        spec_one_volume = []
        if(type_idx ==0): # enhancing tumor
            s_volume[s_volume == 2] = 0
            g_volume[g_volume == 2] = 0
            s_volume[s_volume == 1] = 0
            g_volume[g_volume == 1] = 0
            temp_spec = binary_spec3d(s_volume > 0, g_volume > 0)
            spec_one_volume = [temp_spec]
        elif(type_idx ==1): # whole tumor
            temp_spec = binary_spec3d(s_volume > 0, g_volume > 0)
            spec_one_volume = [temp_spec]
        elif(type_idx == 2): # tumor core
            s_volume[s_volume == 2] = 0
            g_volume[g_volume == 2] = 0
            temp_spec = binary_spec3d(s_volume > 0, g_volume > 0)
            spec_one_volume = [temp_spec]
        else:
            for label in [1, 2, 3, 4]: # spec of each class
                temp_spec = binary_spec3d(s_volume == label, g_volume == label)
                spec_one_volume.append(temp_spec)
        spec_all_data.append(spec_one_volume)
    return spec_all_data


def haus_of_brats_data_set(gt_names, seg_names, type_idx):
    assert(len(gt_names) == len(seg_names))
    haus_all_data = []
    for i in range(len(gt_names)):
        g_volume = load_3d_volume_as_array(gt_names[i])
        s_volume = load_3d_volume_as_array(seg_names[i])
        if(type_idx ==0): # enhancing tumor
            s_volume[s_volume == 2] = 0
            g_volume[g_volume == 2] = 0
            s_volume[s_volume == 1] = 0
            g_volume[g_volume == 1] = 0
            temp_haus = binary_haus3d(s_volume > 0, g_volume > 0)
            if(~np.isnan(temp_haus)):
                haus_one_volume = []
		haus_one_volume = [temp_haus]
		haus_all_data.append(haus_one_volume)

        elif(type_idx ==1): # whole tumor
            temp_haus = binary_haus3d(s_volume > 0, g_volume > 0)
            if(~np.isnan(temp_haus)):
                haus_one_volume = []
		haus_one_volume = [temp_haus]
		haus_all_data.append(haus_one_volume)

        elif(type_idx == 2): # tumor core
            s_volume[s_volume == 2] = 0
            g_volume[g_volume == 2] = 0
            temp_haus = binary_haus3d(s_volume > 0, g_volume > 0)
            if(~np.isnan(temp_haus)):	    
                haus_one_volume = []
		haus_one_volume = [temp_haus]
		haus_all_data.append(haus_one_volume)
        else:
            for label in [1, 2, 3, 4]: # haus of each class
                temp_haus = binary_haus3d(s_volume == label, g_volume == label)
                haus_one_volume = []
                haus_one_volume.append(temp_haus)
        
    return haus_all_data



year = 17
s_folder = '/home/AP85890/brats17/smooth_topo_result17'
g_folder = '/home/AP85890/brats17/data/Brats18TrainingData'
patient_names_file = '/home/AP85890/brats17/config17/test_names.txt'

test_types = ['enhancing', 'whole','core','all']
gt_names  = get_ground_truth_names(g_folder, patient_names_file, year)
seg_names = get_segmentation_names(s_folder, patient_names_file)


for type_idx in range(3):
        dice = dice_of_brats_data_set(gt_names, seg_names, type_idx)
        dice = np.asarray(dice)
        dice_mean = dice.mean(axis = 0)
        dice_std  = dice.std(axis  = 0)
        test_type = test_types[type_idx]
        np.savetxt(s_folder + '/dice_{0:}.txt'.format(test_type), dice)
        np.savetxt(s_folder + '/dice_{0:}_mean.txt'.format(test_type), dice_mean)
        np.savetxt(s_folder + '/dice_{0:}_std.txt'.format(test_type), dice_std)
        print('tissue type', test_type)
        if(test_type == 'all'):
            print('tissue label', [1, 2, 3, 4])
        print('dice mean  ', dice_mean)
        print('dice std   ', dice_std)
for type_idx in range(3):
        sens = sens_of_brats_data_set(gt_names, seg_names, type_idx)
        sens = np.asarray(sens)
        sens_mean = sens.mean(axis = 0)
        sens_std  = sens.std(axis  = 0)
        test_type = test_types[type_idx]
        np.savetxt(s_folder + '/sens_{0:}.txt'.format(test_type), sens)
        np.savetxt(s_folder + '/sens_{0:}_mean.txt'.format(test_type), sens_mean)
        np.savetxt(s_folder + '/sens_{0:}_std.txt'.format(test_type), sens_std)
        print('tissue type', test_type)
        if(test_type == 'all'):
            print('tissue label', [1, 2, 3, 4])
        print('sens mean  ', sens_mean)
        print('sens std   ', sens_std)
for type_idx in range(3):
        spec = spec_of_brats_data_set(gt_names, seg_names, type_idx)
        spec = np.asarray(spec)
        spec_mean = spec.mean(axis = 0)
        spec_std  = spec.std(axis  = 0)
        test_type = test_types[type_idx]
        np.savetxt(s_folder + '/spec_{0:}.txt'.format(test_type), spec)
        np.savetxt(s_folder + '/spec_{0:}_mean.txt'.format(test_type), spec_mean)
        np.savetxt(s_folder + '/spec_{0:}_std.txt'.format(test_type), spec_std)
        print('tissue type', test_type)
        if(test_type == 'all'):
            print('tissue label', [1, 2, 3, 4])
        print('spec mean  ', spec_mean)
        print('spec std   ', spec_std)

for type_idx in range(3):
        haus = haus_of_brats_data_set(gt_names, seg_names, type_idx)
        haus = np.asarray(haus)
        haus_mean = haus.mean(axis = 0)
        haus_std  = haus.std(axis  = 0)
        test_type = test_types[type_idx]
        np.savetxt(s_folder + '/haus_{0:}.txt'.format(test_type), haus)
        np.savetxt(s_folder + '/haus_{0:}_mean.txt'.format(test_type), haus_mean)
        np.savetxt(s_folder + '/haus_{0:}_std.txt'.format(test_type), haus_std)
        print('tissue type', test_type)
        if(test_type == 'all'):
            print('tissue label', [1, 2, 3, 4])
        print('haus mean  ', haus_mean)
        print('haus std   ', haus_std)
print('###########################################################################\n\n\n')
    
