"""Independent integer enumeration of the metric's fixed-size footprint."""
import numpy as np
import pytest

from physmorph.metric_splat import fixed_footprint_counts


@pytest.mark.parametrize('res', [1, 2, 17])
def test_duplicate_clipped_and_excluded_centers(res):
    ij = np.array([[0, 0], [0, 0], [res-1, res-1], [res//2, res//2],
                   [-1, 0], [res, res], [-100, 100]], dtype=np.int64)
    valid = (ij >= 0).all(1) & (ij < res).all(1)
    expected = np.zeros((res, res), dtype=np.int64)
    for (i, j), include in zip(ij, valid):
        if include:
            for a in range(i-1, i+2):
                for b in range(j-1, j+2):
                    expected[min(max(a, 0), res-1), min(max(b, 0), res-1)] += 1
    actual = fixed_footprint_counts(ij, valid, res)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64 and actual.sum() == 9 * valid.sum()


@pytest.mark.parametrize('count', [0, 10])
def test_empty_or_all_excluded(count):
    got = fixed_footprint_counts(np.full((count, 2), 10, dtype=np.int64),
                                 np.zeros(count, dtype=bool), 8)
    assert got.shape == (8, 8) and not got.any()


def test_invalid_layout_is_rejected():
    with pytest.raises(ValueError):
        fixed_footprint_counts(np.zeros((3, 3), dtype=np.int64), np.ones(3, bool), 8)
    with pytest.raises(ValueError):
        fixed_footprint_counts(np.zeros((3, 2)), np.ones(3, bool), 8)
    with pytest.raises(ValueError):
        fixed_footprint_counts(np.zeros((3, 2), dtype=np.int64), np.ones(3), 8)
    with pytest.raises(ValueError):
        fixed_footprint_counts(np.zeros((3, 2), dtype=np.int64), np.ones(3, bool), 0)


@pytest.mark.parametrize('dtype', [np.int8, np.int32, np.uint64])
def test_offset_unsafe_index_dtypes_are_rejected(dtype):
    with pytest.raises(ValueError):
        fixed_footprint_counts(np.array([[127,127]],dtype=dtype), np.ones(1,bool), 128)


def test_metric_dispatch_preserves_masks_without_dynamic_histogram(monkeypatch):
    from physmorph import metrics
    rng=np.random.default_rng(87)
    x=rng.uniform(-1.2,1.2,(400,3)).astype(np.float32)
    x[:20]=[-1.,-1.,0.];x[20:40]=[1.,1.,0.]  # duplicated border/outside centers
    cases=[(x,16,0.,0.,1.),(x,128,.7,-.5,1.),(x[:0],8,0.,0.,1.)]
    expected=[metrics._splat_body(*args) for args in cases]
    monkeypatch.setattr(metrics,'is_cuda_execution',lambda:True)  # dispatch only; CPU oracle arithmetic
    def forbidden(*args,**kwargs): raise AssertionError('Dynamic histogram used')
    monkeypatch.setattr(np,'bincount',forbidden)
    for args,want in zip(cases,expected):
        np.testing.assert_array_equal(metrics._splat_body(*args),want)
