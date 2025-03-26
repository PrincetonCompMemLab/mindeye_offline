import numpy as np
import pytest
from utils import filter_and_average_mst

def test_no_mst_images():
    """Test function with no MST images"""
    vox = np.array([[1,2,3], [4,5,6], [7,8,9]])
    vox_image_dict = {0: 'image1.jpg', 1: 'image2.jpg', 2: 'image3.jpg'}
    
    filtered_vox, kept_indices = filter_and_average_mst(vox, vox_image_dict)
    
    np.testing.assert_array_equal(filtered_vox, vox)
    np.testing.assert_array_equal(kept_indices, [0, 1, 2])

def test_single_mst_image_set():
    """Test function with one set of repeated MST images"""
    vox = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
    vox_image_dict = {0: 'image1.jpg', 1: 'MST_pairs/image2.jpg', 2: 'image3.jpg', 3: 'MST_pairs/image2.jpg'}
    
    filtered_vox, kept_indices = filter_and_average_mst(vox, vox_image_dict)
    
    expected_vox = np.array([[1,2,3], [7,8,9], [7,8,9]])
    expected_indices = [0, 1, 2]
    
    np.testing.assert_array_equal(filtered_vox, expected_vox)
    np.testing.assert_array_equal(kept_indices, expected_indices)

def test_multiple_mst_image_sets():
    """Test function with multiple sets of repeated MST images"""
    vox = np.array([[1,2,3], [4,5,6], [7,8,9], [7,8,9], [10,11,12], [12,15,12]])
    vox_image_dict = {
        0: 'image1.jpg', 
        1: 'MST_pairs/image2.jpg', 
        2: 'image3.jpg', 
        3: 'MST_pairs/image2.jpg', 
        4: 'MST_pairs/image4.jpg', 
        5: 'MST_pairs/image4.jpg'
    }
    
    filtered_vox, kept_indices = filter_and_average_mst(vox, vox_image_dict)
    
    expected_vox = np.array([[1,2,3], [5.5, 6.5, 7.5], [7,8,9], [11,13,12]])
    expected_indices = [0, 1, 2, 4]
    
    np.testing.assert_array_equal(filtered_vox, expected_vox)
    np.testing.assert_array_equal(kept_indices, expected_indices)

def test_empty_input():
    """Test function with empty input"""
    vox = np.array([])
    vox_image_dict = {}
    
    filtered_vox, kept_indices = filter_and_average_mst(vox, vox_image_dict)
    
    assert len(filtered_vox) == 0
    assert len(kept_indices) == 0

def test_input_shape():
    """Test that input shape is preserved"""
    vox = np.random.rand(5, 3)
    vox_image_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}
    
    filtered_vox, _ = filter_and_average_mst(vox, vox_image_dict)
    
    assert filtered_vox.shape[1] == vox.shape[1]