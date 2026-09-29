"""Run on hyde06: device bookkeeping, ownership and strict neighbour regressions."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='hyde06 CUDA checks')


def test_fixed_metric_footprint_avoids_dynamic_histogram(monkeypatch):
    from physmorph import metrics
    from physmorph.compute import cuda_execution, cuda_module, to_array, to_host
    x = np.random.default_rng(87).uniform(-1.2, 1.2, (400, 3)).astype(np.float32)
    x[:20] = [-1., -1., 0.]; x[20:40] = [1., 1., 0.]
    cases = [(x, 16, 0., 0., 1.), (x, 128, .7, -.5, 1.), (x[:0], 8, 0., 0., 1.)]
    expected = [metrics._splat_body(*args) for args in cases]
    def forbidden(*args, **kwargs):
        raise AssertionError('Dynamic CUDA histogram used')
    with cuda_execution('cuda:0'):
        monkeypatch.setattr(cuda_module(), 'bincount', forbidden)
        for (cloud, *args), want in zip(cases, expected):
            actual = metrics._splat_body(to_array(cloud), *args)
            np.testing.assert_array_equal(to_host(actual), want)


def test_metric_summary_matches_cpu_without_cpu_neighbors(monkeypatch):
    from physmorph import metrics
    import physmorph.compute as compute
    x = np.random.default_rng(31).uniform(-1, 1, (400, 3)).astype(np.float32)
    frames = [x + np.array([.003 * i, 0., 0.], np.float32) for i in range(7)]
    target = x * 1.03
    reference = metrics.summarize(frames, target, window=2)
    original = compute.KDTree.__init__
    def device_tree(self, *args, **kwargs):
        assert compute.is_cuda_execution(), 'CPU metric neighbor path'
        original(self, *args, **kwargs)
    monkeypatch.setattr(compute.KDTree, '__init__', device_tree)
    actual = metrics.summarize(frames, target, window=2, compute_backend='cuda')
    assert actual.keys() == reference.keys()
    for key in actual:
        if isinstance(actual[key], str):
            assert actual[key] == reference[key]
            continue
        np.testing.assert_allclose(actual[key], reference[key], atol=2e-6, rtol=2e-5,
                                   equal_nan=True, err_msg=key)


@pytest.mark.parametrize('isochoric', [False, True])
def test_small_pin_cohort_assimilation_matches_numpy(isochoric):
    from physmorph.plasticity.assimilation import assimilate_elastic
    from physmorph.compute import cuda_execution, to_host
    rng = np.random.default_rng(44)
    F = (np.eye(3) + rng.normal(0, .12, (511, 3, 3))).astype(np.float32)
    Fp = (np.eye(3) + rng.normal(0, .06, (511, 3, 3))).astype(np.float32)
    expected = assimilate_elastic(F, Fp, eta=1., isochoric=isochoric)
    with cuda_execution('cuda:0'):
        actual = to_host(assimilate_elastic(F, Fp, eta=1., isochoric=isochoric))
    np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=5e-5)


def test_body_lattice_inverse_mapping_matches_cpu():
    from physmorph.pipeline.body_control import BodyControlBasis
    from physmorph.compute import cuda_execution
    x = np.random.default_rng(92).uniform(-3, 3, (1300, 3)).astype(np.float32)
    x[:100] = x[100]
    reference = BodyControlBasis(x, (-1.1, -.9, 1.2), .3, device='cuda:0')
    with cuda_execution('cuda:0'):
        actual = BodyControlBasis(x, (-1.1, -.9, 1.2), .3, device='cuda:0')
        assert actual.n_nodes == reference.n_nodes
        assert torch.equal(actual.idx, reference.idx)
        assert torch.allclose(actual.weights, reference.weights, atol=1e-7, rtol=1e-6)


def test_repeated_pipeline_calls_keep_source_neighbors_and_backends_isolated(monkeypatch):
    import physmorph.compute as compute
    from physmorph.pipeline import run_pipeline, PipelineConfig
    from physmorph.mpm.state import MPMParams
    observed = []
    original = compute.KDTree.query
    def record(self, x, k=1, **kwargs):
        if k == 4:
            observed.append((self.cuda, compute.to_host(self.data)))
        return original(self, x, k=k, **kwargs)
    monkeypatch.setattr(compute.KDTree, 'query', record)
    sources = [np.random.default_rng(seed).uniform(-1, 1, (80, 3)).astype(np.float32)
               for seed in (81, 82, 83)]
    for source, backend in zip(sources, ('cuda', 'cuda', 'legacy')):
        cfg = PipelineConfig(T=3, iters=1, animations=1, loss_res=12, device='cuda:0',
                             compute_backend=backend, lambda_auto=0., ctrl_rprop=True,
                             ctrl_rprop_smooth=True, ctrl_rprop_k=3)
        result = run_pipeline(source, source * 1.2, MPMParams(dx=1., nx=32, ny=32, nz=32), cfg,
                              log=lambda *args: None)
        assert isinstance(result['frames'][-1], np.ndarray)
        assert np.isfinite(result['frames'][-1]).all()
    assert [cuda for cuda, _ in observed] == [True, True, False]
    for (_, actual), expected in zip(observed, sources):
        np.testing.assert_array_equal(actual, expected)
    assert not compute.is_cuda_execution()


def test_tree_radius_padding_and_owned_warp_snapshot():
    import warp as wp
    from scipy.spatial import cKDTree
    from physmorph.compute import cuda_execution, to_array, to_host, KDTree, warp_array, warp_assign
    points = np.random.default_rng(51).normal(size=(80, 3)).astype(np.float32)
    ref = cKDTree(points)
    with cuda_execution('cuda:0'):
        x = to_array(points)
        tree = KDTree(x)
        d, i = tree.query(x[:7], k=90, distance_upper_bound=.8)
        dr, ir = ref.query(points[:7], k=90, distance_upper_bound=.8)
        np.testing.assert_allclose(to_host(d), dr, atol=1e-10)
        np.testing.assert_array_equal(to_host(i), ir)
        counts = tree.query_ball_point(x, .8, return_length=True)
        np.testing.assert_array_equal(to_host(counts), ref.query_ball_point(points, .8, return_length=True))
        w = warp_array(x, wp.vec3, 'cuda:0')
        snapshot = to_array(w, copy=True)
        warp_assign(w, x * 2)
        np.testing.assert_array_equal(to_host(snapshot), points)
        np.testing.assert_array_equal(to_host(to_array(w)), points * 2)


def test_radius_counts_expand_past_initial_capacity_and_include_boundary(monkeypatch):
    from scipy.spatial import cKDTree
    from physmorph.compute import cuda_execution, KDTree, to_host
    points = np.vstack((np.zeros((80, 3)), np.tile([1., 0., 0.], (80, 1)),
                        np.tile([2., 0., 0.], (80, 1))))
    queries = np.array([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])
    radii = np.array([0., 1., .5])
    expected = cKDTree(points).query_ball_point(queries, radii, return_length=True)
    with cuda_execution('cuda:0'):
        tree = KDTree(points)
        def forbidden(*args, **kwargs):
            raise AssertionError('unsafe native CuPy radius kernel')
        monkeypatch.setattr(type(tree.tree), 'query_ball_point', forbidden)
        actual = tree.query_ball_point(queries, radii, return_length=True)
        np.testing.assert_array_equal(to_host(actual), expected)
        assert int(tree.query_ball_point(queries[0], np.inf, return_length=True)) == len(points)
        assert to_host(tree.query_ball_point(queries[:0], 1., return_length=True)).shape == (0,)
        empty = KDTree(np.empty((0, 3)))
        np.testing.assert_array_equal(to_host(empty.query_ball_point(queries, 1., return_length=True)), 0)
        with pytest.raises(ValueError, match='return_length'):
            tree.query_ball_point(queries, 1.)


@pytest.mark.parametrize('count,k', [(0, 3), (1, 4), (20, 30), (50, 9), (4096, 9)])
def test_cuda_knn_never_falls_back_and_always_returns_self(monkeypatch, count, k):
    from physmorph.render import knn_gpu
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU neighbour fallback')
    monkeypatch.setattr(knn_gpu, '_cpu_knn', forbidden)
    x = torch.randn(count, 3, device='cuda:0')
    if count >= 20:
        x[:20] = 0  # more coincident particles than the requested neighbours
    d, i = knn_gpu.knn_self_torch(x, k)
    assert d.is_cuda and i.is_cuda and d.shape == (count, k)
    if count:
        assert torch.equal(i[:, 0], torch.arange(count, device=x.device))
        assert torch.all(d[:, 0] == 0)
        assert torch.all(d[:, 1:] >= d[:, :-1])
        if k > count:
            assert torch.all(torch.isinf(d[:, count:]))
            assert torch.all(i[:, count:] == count)
    with pytest.raises(ValueError):
        knn_gpu.knn_self_torch(x, 0)


def test_covariance_reconstruction_including_half_turns():
    from scipy.spatial.transform import Rotation
    from physmorph.render.covariance_torch import decompose_cov_torch, rotation_to_quaternion
    rng = np.random.default_rng(11)
    a = rng.normal(size=(17000, 3, 3))
    covariance = a @ a.transpose(0, 2, 1) + np.eye(3) * 1e-4
    scales, q = decompose_cov_torch(torch.tensor(covariance, device='cuda:0', dtype=torch.float32))
    quaternion = q.cpu().numpy()
    rot = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix()
    actual = (rot * scales.cpu().numpy()[:, None, :] ** 2) @ rot.transpose(0, 2, 1)
    np.testing.assert_allclose(actual, covariance, atol=5e-6, rtol=3e-5)
    rotations = torch.diag(torch.tensor([1., -1., -1.], device='cuda:0'))[None]
    assert torch.allclose(rotation_to_quaternion(rotations).abs(), rotations.new_tensor([[0., 1., 0., 0.]]))
