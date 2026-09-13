"""Read-only static bodies cannot couple dynamic solve components."""
import numpy as np
import pytest
from motion_engine.contact_engine_gpu_island_solve import partition_pairs


def test_shared_static_and_original_pair_order():
    a=np.array([0,1,0,2]);b=np.array([3,3,1,-1]);mass=np.array([1.,1.,1.,0.])
    order=np.array([3,1,0,2])
    packed,starts,counts=partition_pairs(a,b,mass,order)
    assert [packed[s:s+c].tolist() for s,c in zip(starts,counts)]==[[1,0,2],[3]]
    # Removing the dynamic link separates the two bodies despite the shared static endpoint.
    packed,starts,counts=partition_pairs(a[:2],b[:2],mass,np.array([1,0]))
    assert counts.tolist()==[1,1]


def test_all_static_empty_and_invalid_inputs():
    a=np.array([0,1]);b=np.array([1,0]);order=np.array([1,0])
    assert partition_pairs(a,b,np.zeros(2),order)[2].tolist()==[1,1]
    empty=np.zeros(0,np.int32)
    assert all(len(x)==0 for x in partition_pairs(empty,empty,np.ones(2),empty))
    with pytest.raises(ValueError):partition_pairs(a,b,np.ones(2),np.array([0,0]))
    with pytest.raises(ValueError):partition_pairs(a,np.array([-2,0]),np.ones(2),order)


@pytest.mark.parametrize('bodies', (64, 65))
def test_noncontracting_component_boundary(bodies, tmp_path):
    wp = pytest.importorskip('warp')
    if wp.get_cuda_device_count() == 0:
        pytest.skip('needs a CUDA device')
    from motion_engine.contact_engine_gpu_prepared_noncontracting import PreparedNoncontractingContactEngine
    from motion_engine.contact_engine_gpu_noncontracting_island import NoncontractingIslandContactEngine
    from engine_metrics import DIMS
    import json
    import hashlib

    results = []
    for cls in (PreparedNoncontractingContactEngine, NoncontractingIslandContactEngine):
        engine = cls(dims=DIMS, mu=.5, vit=40, pit=10)
        for i in range(bodies):
            engine.add_body([0., 0., .0999 + .1998*i])
        captures = []
        original = engine._chunk_launch

        def observe(*args):
            calls = []
            launch = wp.launch

            def counted(kernel, *values, **kw):
                calls.append(kernel.key)
                return launch(kernel, *values, **kw)

            wp.launch = counted
            try:
                return original(*args)
            finally:
                wp.launch = launch
                captures.append(len(calls))

        engine._chunk_launch = observe
        arrays = {}
        sizes = []
        for step in range(3):
            engine.step(1/240, substeps=1)
            sizes.append(engine.max_island_pairs)
            state = engine.get_state()
            for key in ('xc', 'Rm', 'vc', 'om'):
                arrays[f'{step}_{key}'] = np.ascontiguousarray(getattr(state, key))
            arrays[f'{step}_force'] = np.ascontiguousarray(engine.contact_forces())
            for key in ('ckeys', 'sbi', 'sbj', 'spA', 'spB', 'sn', 'spen', 'jn', 'jt1', 'jt2', 'jp'):
                arrays[f'{step}_{key}'] = getattr(engine, key).numpy()[:engine.n_contacts_solved].copy()
        assert sizes == [bodies]*3
        assert all(np.isfinite(a).all() for a in arrays.values())
        np.savez_compressed(tmp_path/f'{cls.name}.npz', **arrays)
        results.append((arrays, captures))
    baseline, candidate = results
    assert set(baseline[0]) == set(candidate[0])
    assert all(baseline[0][k].shape == candidate[0][k].shape
               and baseline[0][k].dtype == candidate[0][k].dtype
               and baseline[0][k].tobytes() == candidate[0][k].tobytes() for k in baseline[0])
    assert len(baseline[1]) == len(candidate[1]) > 0
    if bodies == 64:
        assert all(b < a for a, b in zip(baseline[1], candidate[1]))
    else:
        assert baseline[1] == candidate[1]
    (tmp_path/'boundary.json').write_text(json.dumps(dict(bodies=bodies,
        baseline_calls=baseline[1], candidate_calls=candidate[1],
        hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in candidate[0].items()}),
        sort_keys=True, indent=2)+'\n')
