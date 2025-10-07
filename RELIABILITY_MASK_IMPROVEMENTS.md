# Reliability Mask Calculation Improvements

## Summary

Improved the reliability mask calculation to address two critical issues with the original implementation:

1. **All Repeats Averaging**: Now averages across all available repeat presentations (not just first 2)
2. **Train/Test Purity**: Only uses training set images for reliability calculation to prevent test set contamination

## Changes Made

### 1. New Function in `utils.py`

Added `compute_reliability_with_train_test_split()` which:
- Filters pairs to only include those where ALL repeats are in the training set
- Computes pairwise correlations for all unique combinations of repeats
- Averages across all pairwise correlations (not just one pair)
- Returns reliability scores and the filtered pairs used

```python
rels, train_pairs = utils.compute_reliability_with_train_test_split(
    vox=vox, 
    pairs=pairs, 
    train_indices=train_image_indices,
    test_indices=test_image_indices,
    verbose=True
)
```

### 2. QC/Validation Functions

Added two validation functions:

**`validate_reliability_calculation()`**: Comprehensive QC checks including:
- Reliability score statistics (mean, std, range, percentiles)
- Distribution analysis (% positive voxels, NaN counts)
- Train/test purity verification (overlap checks)
- Repeat usage statistics
- Automatic warning detection

**`compare_reliability_methods()`**: Side-by-side comparison of:
- Old method (first 2 repeats only, all pairs)
- New method (all repeats, train-only pairs)
- Correlation between methods
- Mean difference analysis

### 3. Updated Notebook (`main-multisession-3tasks.ipynb`)

Added new cells after the train/test split section:
- Markdown header explaining the improvements
- Cell to compute new reliability scores with QC
- Cell to compare old vs new methods with visualizations
- Cell to assign the new scores to `rels` for downstream use
- Updated old method header with deprecation notice

## How It Works

### Old Method Issues

```python
# Only uses first 2 repeats
pairs_homog = np.array([[p[0], p[1]] for p in pairs])

# No train/test filtering - uses ALL pairs
vox_pairs = utils.zscore(vox[pairs_homog])
rels = np.full(vox.shape[-1],np.nan)
for v in tqdm(range(vox.shape[-1])):
    rels[v] = np.corrcoef(vox_pairs[:,0,v], vox_pairs[:,1,v])[1,0]
```

**Problems:**
1. If an image has 3+ repeats, only uses first 2 → wastes information
2. Includes test set images in reliability calculation → contamination
3. No verification of train/test purity

### New Method Solution

```python
# Filter to train-only pairs
train_pairs = []
for pair in pairs:
    if all(idx in train_set for idx in pair):
        train_pairs.append(pair)

# For each voxel, compute correlations for ALL pairwise combinations
for v in range(n_voxels):
    all_corrs = []
    for pair in train_pairs:
        pair_responses = vox[pair, v]
        # Z-score within this pair
        pair_responses = zscore(pair_responses)
        
        # All unique pairs of repeats
        combos = list(itertools.combinations(range(len(pair)), 2))
        for i, j in combos:
            r = np.corrcoef(pair_responses[i], pair_responses[j])[0, 1]
            all_corrs.append(r)
    
    # Average across all correlations
    rels[v] = np.mean(all_corrs)
```

**Benefits:**
1. Uses ALL available repeats → more robust estimates
2. Train/test pure → no contamination
3. Automatically validated with QC checks

## Usage Example

### Running the Improved Calculation

In the notebook, after defining train/test splits:

```python
# Compute reliability with train/test purity
rels_new, train_pairs_used = utils.compute_reliability_with_train_test_split(
    vox=vox, 
    pairs=pairs, 
    train_indices=train_image_indices,
    test_indices=test_image_indices,
    verbose=True
)

# Run QC validation
qc_results = utils.validate_reliability_calculation(
    rels=rels_new,
    vox=vox,
    pairs=train_pairs_used,
    train_indices=train_image_indices,
    test_indices=test_indices
)

# Compare with old method
comparison, rels_old, _ = utils.compare_reliability_methods(
    vox=vox,
    pairs=pairs,
    train_indices=train_image_indices,
    test_indices=test_image_indices
)

# Use the new reliability scores
rels = rels_new
```

### Expected QC Output

```
==================================================
RELIABILITY QC SUMMARY
==================================================
Reliability Statistics:
  Mean ± Std: 0.XXXX ± 0.XXXX
  Median: 0.XXXX
  Range: [-X.XXXX, X.XXXX]
  Q25-Q75: [0.XXXX, 0.XXXX]
  % Positive: XX.X%
  NaN voxels: X

Repeat Usage:
  Total pairs: XXX
  Repeats per pair: 2-X (mean: X.X)

Train/Test Purity:
  Train/test overlap: 0 indices
  Pure split: ✓ PASS
  Test contamination in pairs: 0 indices

Warnings:
  ✓ All checks passed
==================================================
```

## Next Steps for Testing

To validate on sub-005 ses-03:

1. **Set parameters in notebook:**
   ```python
   sub = 'sub-005'
   session = 'ses-03'
   train_test_split = 'MST'  # or whatever split you're using
   ```

2. **Run through the notebook** up to the new reliability calculation cells

3. **Check QC output:**
   - Verify reliability mean is positive and reasonable (typically 0.1-0.5)
   - Confirm train/test purity passes (0 overlap)
   - Check that multiple repeats are being used (mean > 2.0)
   - Compare correlation between old and new methods (should be high, > 0.8)

4. **Visual inspection:**
   - Histogram should show reasonable distribution
   - Scatter plot should show strong correlation with old method
   - Difference plot should be centered near 0

5. **Launch training** with the new reliability mask

## Implementation Details

### Handling Edge Cases

- **No train pairs**: Returns NaN array if no pairs have all repeats in training set
- **Mixed repeat counts**: Handles pairs with different numbers of repeats (2, 3, 4, etc.)
- **NaN handling**: Skips NaN correlations when averaging

### Performance Considerations

- Uses tqdm for progress tracking (can disable with `verbose=False`)
- Computes correlations on-the-fly rather than storing all responses
- Memory efficient for large voxel counts

### Z-scoring Strategy

Z-scoring is done **per pair** before computing correlations:
- Normalizes for differences in overall activation levels
- Makes correlations comparable across pairs
- Consistent with original implementation

## Testing Checklist

Before launching full training:

- [ ] Function returns expected shape
- [ ] No NaN values in reliability scores (unless expected)
- [ ] Train/test purity check passes (0 overlap)
- [ ] Mean reliability is positive and reasonable
- [ ] High correlation with old method (> 0.8)
- [ ] Visualizations look reasonable
- [ ] All available repeats are being used
- [ ] QC warnings pass

## Future Improvements

For handling multiple sessions with inhomogeneous repeats:

1. **Session-specific reliability**: Compute reliability per session, then aggregate
2. **Weighted averaging**: Weight by number of repeats available
3. **Cross-session validation**: Use one session's reliability to validate another
4. **Adaptive thresholding**: Different thresholds per session based on repeat counts

## References

- Original `compute_avg_repeat_corrs()` function (line 826 in utils.py)
- Notebook section: "Reliability calculation" (cells #VSC-bfa240ac onwards)
- Train/test split logic (cell #VSC-99479ec0)
