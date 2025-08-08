import numpy as np
import pytest
from utils import filter_and_average_mst, verify_image_patterns, compute_vox_rels, compute_avg_repeat_corrs

# === filter_and_average_mst tests ===

def test_no_mst_images():
    vox = np.array([[1,2,3], [4,5,6], [7,8,9]])
    vox_image_dict = {0: 'image1.jpg', 1: 'image2.jpg', 2: 'image3.jpg'}
    
    filtered_vox, kept_indices = filter_and_average_mst(vox, vox_image_dict)
    
    np.testing.assert_array_equal(filtered_vox, vox)
    np.testing.assert_array_equal(kept_indices, [0, 1, 2])

def test_single_mst_image_set():
    vox = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
    vox_image_dict = {0: 'image1.jpg', 1: 'MST_pairs/image2.jpg', 2: 'image3.jpg', 3: 'MST_pairs/image2.jpg'}
    
    filtered_vox, kept_indices = filter_and_average_mst(vox, vox_image_dict)
    
    expected_vox = np.array([[1,2,3], [7,8,9], [7,8,9]])
    expected_indices = [0, 1, 2]
    
    np.testing.assert_array_equal(filtered_vox, expected_vox)
    np.testing.assert_array_equal(kept_indices, expected_indices)

def test_multiple_mst_image_sets():
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
    vox = np.array([])
    vox_image_dict = {}
    
    filtered_vox, kept_indices = filter_and_average_mst(vox, vox_image_dict)
    
    assert len(filtered_vox) == 0
    assert len(kept_indices) == 0

def test_input_shape():
    vox = np.random.rand(5, 3)
    vox_image_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}
    
    filtered_vox, _ = filter_and_average_mst(vox, vox_image_dict)
    
    assert filtered_vox.shape[1] == vox.shape[1]


# === verify_image_patterns tests ===

def test_valid_special515():
    image_to_indices = {
        "all_stimuli/special515/image1.jpg": [[1, 2, 3], []],
        "all_stimuli/special515/image2.jpg": [[], [10, 11, 12]],
    }
    failures = verify_image_patterns(image_to_indices)
    assert failures == []

def test_invalid_special515():
    image_to_indices = {
        "all_stimuli/special515/image1.jpg": [[1, 2], []],
        "all_stimuli/special515/image2.jpg": [[1, 2], [3]],
    }
    failures = verify_image_patterns(image_to_indices)
    assert len(failures) == 2

def test_valid_MST_pairs():
    image_to_indices = {
        "all_stimuli/MST_pairs/image1.png": [[4, 5], [6, 7]],
    }
    failures = verify_image_patterns(image_to_indices)
    assert failures == []

def test_invalid_MST_pairs():
    image_to_indices = {
        "all_stimuli/MST_pairs/image1.png": [[4, 5, 6], [7]],
    }
    failures = verify_image_patterns(image_to_indices)
    assert len(failures) == 1

def test_valid_other_images():
    image_to_indices = {
        "all_stimuli/other/image1.png": [[123], []],
        "all_stimuli/other/image2.png": [[], [456]],
    }
    failures = verify_image_patterns(image_to_indices)
    assert failures == []

def test_invalid_other_images():
    image_to_indices = {
        "all_stimuli/other/image1.png": [[123, 124], []],
        "all_stimuli/other/image2.png": [[123], [456]],
    }
    failures = verify_image_patterns(image_to_indices)
    assert len(failures) == 2


# === compute_vox_rels tests ===

# def test_reliability_two_repeats():
#     np.random.seed(0)
#     vox = np.random.rand(70, 10)  # 50 trials, 10 voxels
#     pairs = [
#         [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14],
#         [15, 16, 17], [18, 19, 20], [21, 22, 23], [24, 25, 26], [27, 28, 29],
#         [30, 31, 32], [33, 34, 35], [36, 37, 38], [39, 40, 41], [42, 43, 44],
#         [45, 46, 47], [48, 49, 50], [51, 52, 53], [54, 55, 56], 
#         [57, 58, 59, 60], [61, 62, 63, 64]
#     ]

#     rels = compute_vox_rels(vox, pairs, "sub-01", "ses-01")

#     assert rels.shape == (10,)
#     assert not np.all(np.isnan(rels)), "All voxel reliabilities are NaN!"
#     assert np.all((rels >= -1) & (rels <= 1))


# def test_reliability_three_repeats():
#     np.random.seed(1)
#     vox = np.random.rand(15, 3)  # 15 trials, 3 voxels
#     pairs = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

#     rels = compute_vox_rels(vox, pairs, "sub-01", "ses-02")

#     assert rels.shape == (3,)
#     assert not np.all(np.isnan(rels)), "All voxel reliabilities are NaN!"
#     assert np.all((rels >= -1) & (rels <= 1))


# def test_reliability_four_repeats_mixed():
#     np.random.seed(2)
#     vox = np.random.rand(20, 4)  # 20 trials, 4 voxels
#     pairs = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]  # includes 2 and 4 repeats

#     rels = compute_vox_rels(vox, pairs, "sub-test", "ses-test")

#     assert rels.shape == (4,)
#     assert not np.all(np.isnan(rels)), "All voxel reliabilities are NaN!"
#     assert np.all((rels >= -1) & (rels <= 1))


# def test_near_uniform_data():
#     np.random.seed(42)
#     # Add very small noise to a constant baseline
#     vox = np.ones((6, 3)) + np.random.normal(0, 1e-5, (6, 3))
#     pairs = [[0, 1], [2, 3], [4, 5]]

#     rels = compute_vox_rels(vox, pairs, "sub-near-uniform", "ses-01")

#     assert rels.shape == (3,)
#     assert not np.all(np.isnan(rels)), "All voxel reliabilities are NaN!"
#     assert np.all((rels >= -1) & (rels <= 1))

# def test_invalid_pairs_length():
#     vox = np.random.rand(10, 3)
#     pairs = [[0]]  # should raise due to too few repeats

#     with pytest.raises(AssertionError):
#         compute_vox_rels(vox, pairs, "sub-err", "ses-01")

        
def test_basic_case():
    """Test with 2 repeats and 2 voxels, with basic correlation"""
    vox_repeats = np.random.rand(30, 50)
    breakpoint()
    rels = compute_avg_repeat_corrs(vox_repeats)
    
    # Expected correlation for each voxel should be the correlation between repeat 0 and repeat 1
    assert rels.shape == (2,)  # Should return a vector of size 2 (one per voxel)
    
    # Check that the correlation is valid and close to expected value
    assert np.all(np.isfinite(rels))  # Ensure no NaNs in the results
    
    for v in range(2):  # Check correlation for each voxel
        expected_corr = np.corrcoef(vox_repeats[:, v])[0, 1]
        assert np.allclose(rels[v], expected_corr, atol=1e-5)  # Allow for floating point errors

def test_multiple_repeats():
    """Test with more repeats (3) and multiple voxels (3)"""
    vox_repeats = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])  # 3 repeats, 3 voxels
    rels = compute_avg_repeat_corrs(vox_repeats)
    
    assert rels.shape == (3,)  # Should return a vector of size 3 (one per voxel)
    for v in range(3):
        assert not np.isnan(rels[v])  # Ensure no NaNs are present

def test_identical_repeats():
    """Test with all identical repeats (perfect correlation)"""
    vox_repeats = np.array([[1, 1], [1, 1]])  # Identical repeats, 2 voxels
    rels = compute_avg_repeat_corrs(vox_repeats)
    
    assert rels.shape == (2,)
    assert np.allclose(rels, 1)  # Perfect correlation (should be 1 for all voxels)

def test_anticorrelation():
    """Test with perfect anti-correlation (correlation = -1)"""
    vox_repeats = np.array([[1, 2], [2, 1]])  # Perfect anti-correlation between repeats
    rels = compute_avg_repeat_corrs(vox_repeats)
    
    assert rels.shape == (2,)
    assert np.allclose(rels, -1)  # Perfect negative correlation

def test_zero_variance_repeats():
    """Test with repeats having zero variance (e.g., all values are the same)"""
    vox_repeats = np.array([[1, 1], [1, 1], [1, 1]])  # Zero variance across repeats
    rels = compute_avg_repeat_corrs(vox_repeats)
    
    assert rels.shape == (2,)
    # Since variance is zero, the correlation will be NaN
    assert np.all(np.isnan(rels))

def test_edge_case_two_repeats_and_one_voxel():
    """Test with only 2 repeats and 1 voxel (minimal edge case)"""
    vox_repeats = np.array([[1], [2]])  # 2 repeats, 1 voxel
    rels = compute_avg_repeat_corrs(vox_repeats)
    
    assert rels.shape == (1,)
    assert np.allclose(rels[0], np.corrcoef([1], [2])[1, 0])  # Correlation between the two repeats
