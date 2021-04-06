import os
import numpy as np
import json
from scipy.linalg import norm
from PIL import Image


def get_ref_red_lights():
    """Get some reference images of red lights from the first image."""
    coords_dict = {
        0: [[154, 316, 171, 323], [180, 67, 205, 79], [192, 419, 207, 428]],
        1: [[175, 322, 197, 332], [215, 44, 245, 59], [222, 400, 245, 410]],
        2: [[232, 121, 255, 129], [199, 278, 219, 292], [202, 335, 220, 342], [243, 414, 265, 423]],
        9: [[13, 122, 85, 174], [25, 320, 94, 350], [174, 600, 241, 629]]
    }
    ref = []
    for i in coords_dict:
        I = Image.open(os.path.join(data_path,file_names[i]))
        I = np.asarray(I)
        for coords_list in coords_dict[i]:
            tl_row, tl_col, br_row, br_col = top_row, left_col, bot_row, right_col = coords_list
            obj = I[top_row:bot_row, left_col:right_col, :]
            ref.append(obj)
    return ref


def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    
    dists_rgb = []
    dists_tb = []
    for ref in ref_list:
        ref_height, ref_width, _ = ref.shape
        box_height = ref_height
        box_width = ref_width
        (n_rows,n_cols,n_channels) = np.shape(I)
        for i in range(n_rows - box_height):
            for j in range(n_cols - box_width):
                tl_row = i
                br_row = i + box_height
                tl_col = j
                br_col = j + box_width
                test_box = I[tl_row:br_row, tl_col:br_col, :]
                test_box_max = np.max(test_box)
                test_box_min = np.min(test_box)
                if test_box_max < 150 or test_box_min > 100:
                    continue
                dist_rgb = norm(ref - test_box)

                if dist_rgb < 1000:
                    bounding_boxes.append([tl_row,tl_col,br_row,br_col])
                if dist_rgb < 1500:
                    test_box_mean = np.mean(test_box)
                    test_box_std = np.std(test_box)
                    if test_box_max > 150 and test_box_max > test_box_mean + test_box_std*2:
                        if tl_row + (br_col - tl_col)*2 < br_row:
                            a = I[tl_row:tl_row + (br_col - tl_col), tl_col:br_col, :]
                            b = I[tl_row + (br_col - tl_col):tl_row + (br_col - tl_col)*2,
                                  tl_col:br_col, :]
                            dist_tb = norm(a - b)
                            if dist_tb > 7000:
                                bounding_boxes.append([tl_row,tl_col,br_row,br_col])
                        else:
                            bounding_boxes.append([tl_row,tl_col,br_row,br_col])
    
    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

user_profile = os.environ['HOME']

# set the path to the downloaded data: 
data_path = '%s/data/EE148/RedLights2011_Medium' % user_profile

# set a path for saving predictions: 
preds_path = '%s/data/EE148/hw01_preds/' % user_profile
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

ref_list = get_ref_red_lights()
preds = {}
for i in range(len(file_names)):
    
    print(i)
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
