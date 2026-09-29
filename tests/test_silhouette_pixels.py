import numpy as np
import pytest

from physmorph.metrics import sil_iou, target_extent
from scripts.probes.silhouette_pixels import transitions, supplier_ids, analyze


def test_transitions_keep_false_positives_and_holes_separate():
    base=np.ones((5,5),bool); candidate=base.copy();candidate[2,2]=False
    target=base.copy()
    got=transitions(base,candidate,target)
    assert got == dict(lost_tp=1,gained_tp=0,gained_fp=0,removed_fp=0,
                       new_internal_hole_pixels=1,removed_internal_hole_pixels=0)
    base=np.zeros((5,5),bool);candidate=base.copy();candidate[0,0]=True
    target=base.copy()
    assert transitions(base,candidate,target)['gained_fp']==1
    assert transitions(candidate,base,target)['removed_fp']==1


def test_supplier_ids_unique_despite_clipped_duplicate_taps():
    ij=np.array([[0,0],[0,0],[1,1],[2,0],[-1,0],[7,7]])
    valid=(ij>=0).all(1)&(ij<8).all(1)
    np.testing.assert_array_equal(supplier_ids(ij,valid,np.array([0,0])),[0,1,2])
    np.testing.assert_array_equal(supplier_ids(ij,valid,np.array([7,7])),[5])


def test_saved_endpoints_close_and_supplier_csr_binds_actual_positions():
    cloud=np.array([[0,0,0],[.9,.9,.2],[-.7,.6,-.3],[.3,-.8,.6]],np.float32)
    clouds=[cloud.copy() for _ in range(6)];clouds[-1][1,0]+=.15
    obs=dict(x0=cloud,target=cloud,pins=np.array([1,0,0,0],bool),start_arrived=np.ones(4,bool))
    extent=target_extent(cloud)
    rows=[dict(geometry=dict(sil_iou=sil_iou(x,cloud,extent,res=8))) for x in clouds]
    summary,data=analyze(obs,clouds,rows,res=8)
    assert summary['baseline_masks_identical'] and summary['exact_count_mask_parity']
    assert summary['changed_view_pixels']>0
    assert len(data['supplier_ptr'])==len(data['pixels'])*6+1
    assert data['supplier_ptr'][-1]==len(data['supplier_ids'])
    np.testing.assert_array_equal(data['material_positions'],np.stack(clouds)[:,data['material_ids']])
    rows[-1]['geometry']['sil_iou']-=.1
    with pytest.raises(ValueError,match='IoU failed closure'): analyze(obs,clouds,rows,res=8)
    clouds[-1][0,0]+=.1
    with pytest.raises(ValueError,match='Pinned endpoint'): analyze(obs,clouds,rows,res=8)


def test_identical_endpoints_have_empty_supplier_table():
    x=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    obs=dict(x0=x,target=x,pins=np.zeros(3,bool),start_arrived=np.ones(3,bool))
    summary,data=analyze(obs,[x]*6,[dict(geometry=dict(sil_iou=1.))]*6,res=8)
    assert summary['changed_view_pixels']==0 and summary['material_ids']==[]
    assert data['supplier_ptr'].tolist()==[0] and data['pixels'].shape==(0,3)
