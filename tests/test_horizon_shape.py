import numpy as np
import pytest

from scripts.probes.horizon_shape import ShapeObserver, physical_frame_indices, sample_schedule, pack_mask_rows


def test_frame_selection_excludes_null_holds_and_checks_overlap():
    assert physical_frame_indices([dict(start_frame=0,end_frame=2),
                                   dict(start_frame=3,end_frame=5)]) == [0,1,2,4,5]
    with pytest.raises(ValueError,match='overlapping'):
        physical_frame_indices([dict(start_frame=0,end_frame=2),dict(start_frame=1,end_frame=3)])


def hole_fixture():
    # A 5x5 planar lattice: a missing center opens a projected interior hole.
    xy=np.array([(x,y) for x in range(-2,3) for y in range(-2,3)],np.float32)
    target=np.column_stack((xy,np.zeros(len(xy),np.float32)))
    target=np.repeat(target,3,axis=0)
    target[::3,2]=-.1;target[2::3,2]=.1
    middle=target.copy()
    missing=(middle[:,0]==0)&(middle[:,1]==0)
    middle[missing]=middle[0]
    return target,middle


def test_every_phase_finds_transient_hole_hidden_by_equal_endpoints():
    target,middle=hole_fixture()
    observer=ShapeObserver(target,resolutions=(16,),views=[(0.,0.)])
    before,_=observer.frame(target)
    transient,packed=observer.frame(middle)
    after,_=observer.frame(target)
    a,b,c=(row['views']['16'] for row in (before,transient,after))
    assert a['internal_hole_pixels']==c['internal_hole_pixels']==[0]
    assert b['internal_hole_pixels'][0]>0
    assert b['new_hole_pixels'][0]>0 and c['removed_hole_pixels'][0]>0
    assert transient['target_covered']<before['target_covered']
    assert before['target_covered']==after['target_covered']==len(target)
    assert np.unpackbits(packed['body_16_bits'],axis=1).shape==(1,256)


def test_target_extent_does_not_rescale_for_ejecta():
    target,_=hole_fixture()
    observer=ShapeObserver(target,resolutions=(16,),views=[(0.,0.)])
    before=observer.extent
    x=target.copy();x[0]=100
    row,_=observer.frame(x)
    assert observer.extent==before and row['source_outside_extent_box']==1
    assert row['views']['16']['projected_outside_centers']==[1]


def test_raw_terminal_sample_is_separate_at_same_time_not_an_archive_frame():
    samples=sample_schedule([dict(animation=0,start_frame=0,end_frame=2),
                             dict(animation=3,start_frame=3,end_frame=5)])
    assert [s['archive_frame'] for s in samples]==[0,1,None,2,4,None,5]
    assert [s['paired_archive_frame'] for s in samples]==[0,1,2,2,4,5,5]
    assert samples[2]['kind']=='optimizer_raw_endpoint'
    assert samples[3]['kind']=='promoted_endpoint'


def test_packed_mask_rows_preserve_independent_zero_padded_tail_bits():
    masks=np.array([[[True,False,True],[False,False,True],[True,False,True]],
                    [[False,True,False],[True,True,False],[False,True,False]]])
    np.testing.assert_array_equal(pack_mask_rows(masks),np.packbits(masks.reshape(2,-1),axis=1))
