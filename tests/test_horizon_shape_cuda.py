import numpy as np
import pytest
import torch

from physmorph.compute import cuda_execution, to_array, to_host
from scripts.probes.horizon_shape import ShapeObserver
from test_horizon_shape import hole_fixture

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='hyde06 CUDA checks')


def test_cuda_all_phase_holes_coverage_bits_and_clipping_match_cpu():
    target,middle=hole_fixture()
    cpu=ShapeObserver(target,resolutions=(16,32),views=[(0.,0.),(.7,.3)])
    ejecta=target.copy();ejecta[0]=100.
    rotated=target.copy();rotated[0]=[.9*cpu.extent,0.,-.9*cpu.extent]
    frames=(target,middle,target,ejecta,rotated)
    reference=[cpu.frame(x) for x in frames]
    assert reference[3][0]['source_outside_extent_box']==1
    assert reference[4][0]['source_outside_extent_box']==0
    assert reference[4][0]['views']['16']['projected_outside_centers']==[0,1]
    with cuda_execution('cuda:0'):
        gpu=ShapeObserver(to_array(target),resolutions=(16,32),views=[(0.,0.),(.7,.3)])
        assert hasattr(gpu.target,'__cuda_array_interface__')
        actual=[gpu.frame(to_array(x)) for x in frames]
        for (row,bits),(want,want_bits) in zip(actual,reference):
            assert row==want
            for key,value in bits.items():
                np.testing.assert_array_equal(to_host(value),want_bits[key],err_msg=key)
