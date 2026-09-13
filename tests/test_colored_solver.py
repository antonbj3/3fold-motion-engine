"""Graph-coloured GPU contact solver: the contract checks of its selftest, as pytest, for the plain
coloured solve and for the manifold-reduced one.

Needs a CUDA device. The engine is exercised through the ContactEngine Protocol only, at its shipped
default iteration counts, and the gates are the selftest's own.
"""
import numpy as np
import pytest

from motion_engine.contact_engine import ContactEngine

G = 9.81


def _cuda_available():
    try:
        import warp as wp
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


cuda_only = pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA device")


def _engine(**kw):
    from motion_engine.contact_engine_gpu_colored import GraphColoredContactEngine
    return GraphColoredContactEngine(**kw)


@cuda_only
def test_satisfies_the_contact_engine_protocol():
    assert isinstance(_engine(), ContactEngine)


@cuda_only
def test_k8_stack_is_stable_and_keeps_its_spacing():
    e = _engine()
    for k in range(8):
        e.add_body([0, 0, 0.101 + k * 0.205])
    for _ in range(400):
        e.step(1 / 240)
    st = e.get_state()
    ke = float(np.sum(0.5 * e._M[:, None] * st.vc ** 2))
    zs = sorted(st.xc[:, 2])
    seps = [zs[i + 1] - zs[i] for i in range(7)]
    assert ke < 0.5, f"kinetic energy {ke}"
    assert all(0.15 < sp < 0.25 for sp in seps), f"spacings {seps}"
    assert e.n_colors > 0 and e.n_contacts > 0


@cuda_only
def test_contact_force_on_a_box_at_rest_is_Mg():
    e = _engine()
    e.add_body([0, 0, 0.101])
    for _ in range(200):
        e.step(1 / 240)
    Fz = e.contact_forces()[0, 2]
    Mg = e._M[0] * G
    assert abs(Fz - Mg) / Mg * 100 < 8, f"Fz={Fz} Mg={Mg}"


@cuda_only
def test_two_runs_are_bit_identical():
    def run():
        e = _engine()
        for k in range(4):
            e.add_body([0, 0, 0.101 + k * 0.205])
        for _ in range(200):
            e.step(1 / 240)
        return e.get_state().xc
    assert float(np.max(np.abs(run() - run()))) == 0.0


@cuda_only
def test_manifold_reduction_keeps_the_stack_and_cuts_the_colours():
    """manifold_reduce=True: at most 4 contacts per body pair, colour count down to 8 on the K=8 stack,
    same stability, and the reduction is the only difference to the plain coloured solve."""
    e = _engine(manifold_reduce=True)
    for k in range(8):
        e.add_body([0, 0, 0.101 + k * 0.205])
    for _ in range(400):
        e.step(1 / 240)
    st = e.get_state()
    ke = float(np.sum(0.5 * e._M[:, None] * st.vc ** 2))
    zs = sorted(st.xc[:, 2])
    seps = [zs[i + 1] - zs[i] for i in range(7)]
    assert ke < 0.5, f"kinetic energy {ke}"
    assert all(0.15 < sp < 0.25 for sp in seps), f"spacings {seps}"
    assert e.n_colors <= 8, f"colours {e.n_colors}"
    assert e.n_contacts_solved < e.n_contacts
    assert e.n_contacts_solved <= 4 * 8, f"solved {e.n_contacts_solved} for 8 body pairs at most"


@cuda_only
def test_manifold_reduction_conserves_the_contact_force():
    e = _engine(manifold_reduce=True)
    e.add_body([0, 0, 0.101])
    for _ in range(200):
        e.step(1 / 240)
    Fz = e.contact_forces()[0, 2]
    Mg = e._M[0] * G
    assert abs(Fz - Mg) / Mg * 100 < 1, f"Fz={Fz} Mg={Mg}"


@cuda_only
def test_manifold_reduction_is_bit_identical_over_two_runs():
    def run():
        e = _engine(manifold_reduce=True)
        for k in range(4):
            e.add_body([0, 0, 0.101 + k * 0.205])
        for _ in range(200):
            e.step(1 / 240)
        return e.get_state().xc
    assert float(np.max(np.abs(run() - run()))) == 0.0


@cuda_only
def test_manifold_selection_on_device_equals_the_host_selection():
    """Step 3a: the grouping, the E-optimal selection and the warm start run as GPU kernels by default
    (`manifold_on_host=False`). The device stage must keep the same points as the numpy stage, in the same
    order, on the K=8 stack and on the 1e3 lattice, and the states must stay bit-identical."""
    def pair(centres, steps):
        eh = _engine(manifold_reduce=True, manifold_on_host=True)
        ed = _engine(manifold_reduce=True)
        for c in centres:
            eh.add_body(c)
            ed.add_body(c)
        for _ in range(steps):
            eh.step(1 / 240)
            ed.step(1 / 240)
            assert eh.n_contacts_solved == ed.n_contacts_solved, \
                f"solved {eh.n_contacts_solved} on the host vs {ed.n_contacts_solved} on the device"
            assert np.array_equal(eh.selection_debug(), ed.selection_debug()), "kept-point sets differ"
        assert float(np.max(np.abs(eh.get_state().xc - ed.get_state().xc))) == 0.0

    pair([[0, 0, 0.101 + k * 0.205] for k in range(8)], 120)
    side = 23
    lat = [[ix * 0.35, iy * 0.35, 0.101 + lz * 0.205]
           for lz in range(2) for iy in range(side) for ix in range(side)][:1000]
    pair(lat, 12)


@cuda_only
def test_device_manifold_reaches_the_same_stack_and_force_as_the_host_stage():
    e = _engine(manifold_reduce=True)          # device path, the default
    e.add_body([0, 0, 0.101])
    for _ in range(200):
        e.step(1 / 240)
    Fz = e.contact_forces()[0, 2]
    Mg = e._M[0] * G
    assert abs(Fz - Mg) / Mg * 100 < 1, f"Fz={Fz} Mg={Mg}"
    assert e.n_contacts_solved <= 4


@cuda_only
def test_graph_captured_solve_is_bit_identical_to_the_uncaptured_solve():
    """Step 3b: with `graph_capture=True` the whole solve loop (every colour x every iteration) is replayed
    from a CUDA graph and the colour offsets and counts are read from device arrays. It is the same solve, so
    the states must stay bit-identical to the launch-per-colour path, step by step, on the K=8 stack and on
    the 1e3 lattice, with the same contacts solved."""
    def pair(centres, steps, **kw):
        eu = _engine(manifold_reduce=True, **kw)
        eg = _engine(manifold_reduce=True, graph_capture=True, **kw)
        for c in centres:
            eu.add_body(c)
            eg.add_body(c)
        for _ in range(steps):
            eu.step(1 / 240)
            eg.step(1 / 240)
            assert eu.n_contacts_solved == eg.n_contacts_solved, \
                f"solved {eu.n_contacts_solved} uncaptured vs {eg.n_contacts_solved} captured"
            assert eu.n_colors == eg.n_colors, f"colours {eu.n_colors} vs {eg.n_colors}"
        assert float(np.max(np.abs(eu.get_state().xc - eg.get_state().xc))) == 0.0
        assert eg.n_graph_captures > 0

    pair([[0, 0, 0.101 + k * 0.205] for k in range(8)], 400)
    side = 23
    lat = [[ix * 0.35, iy * 0.35, 0.101 + lz * 0.205]
           for lz in range(2) for iy in range(side) for ix in range(side)][:1000]
    pair(lat, 12)


@cuda_only
def test_the_two_graph_modes_agree():
    """graph_mode='device' (offsets from device arrays, one graph per colour count) and graph_mode='spans'
    (offsets baked in, re-captured when the partition changes) are the same solve."""
    def run(mode):
        e = _engine(manifold_reduce=True, graph_capture=True, graph_mode=mode)
        for k in range(8):
            e.add_body([0, 0, 0.101 + k * 0.205])
        for _ in range(200):
            e.step(1 / 240)
        return e
    a, b = run("device"), run("spans")
    assert float(np.max(np.abs(a.get_state().xc - b.get_state().xc))) == 0.0
    assert a.n_graph_captures <= b.n_graph_captures


@cuda_only
def test_pair_chunks_colour_the_pair_graph_and_stay_bit_identical():
    """Step 3c: with `pair_chunks=True` the colouring runs on the body-pair graph and every pair's contacts
    are solved as one sequential Gauss-Seidel chunk inside one thread. Gates: fewer colours than the
    per-contact colouring on the K=8 stack, one pair per contacting body pair, and two runs bit-identical."""
    def run(**kw):
        e = _engine(manifold_reduce=True, graph_capture=True, pair_chunks=True, **kw)
        for k in range(8):
            e.add_body([0, 0, 0.101 + k * 0.205])
        for _ in range(200):
            e.step(1 / 240)
        return e
    a = run()
    b = run()
    assert float(np.max(np.abs(a.get_state().xc - b.get_state().xc))) == 0.0
    assert a.n_pairs == 8, f"pairs {a.n_pairs}"
    assert a.n_colors < 8, f"colours {a.n_colors} not fewer than the per-contact colouring's 8"
    assert a.n_contacts_solved <= 4 * a.n_pairs


@cuda_only
def test_pair_chunks_in_registers_equal_the_chunk_through_global_memory():
    """The two chunk modes are the same sequential sweep: `global` calls the per-contact warp function once
    per contact of the pair, `registers` keeps the two bodies' velocities in registers over the chunk."""
    def run(mode):
        e = _engine(manifold_reduce=True, graph_capture=True, pair_chunks=True, chunk_mode=mode)
        for k in range(8):
            e.add_body([0, 0, 0.101 + k * 0.205])
        for _ in range(200):
            e.step(1 / 240)
        return e.get_state().xc
    assert float(np.max(np.abs(run("registers") - run("global")))) == 0.0


@cuda_only
def test_pair_chunks_keep_the_stack_and_the_contact_force():
    e = _engine(manifold_reduce=True, graph_capture=True, pair_chunks=True)
    for k in range(8):
        e.add_body([0, 0, 0.101 + k * 0.205])
    for _ in range(400):
        e.step(1 / 240)
    st = e.get_state()
    ke = float(np.sum(0.5 * e._M[:, None] * st.vc ** 2))
    zs = sorted(st.xc[:, 2])
    seps = [zs[i + 1] - zs[i] for i in range(7)]
    assert ke < 0.5, f"kinetic energy {ke}"
    assert all(0.15 < sp < 0.25 for sp in seps), f"spacings {seps}"

    f = _engine(manifold_reduce=True, graph_capture=True, pair_chunks=True)
    f.add_body([0, 0, 0.101])
    for _ in range(200):
        f.step(1 / 240)
    Fz = f.contact_forces()[0, 2]
    Mg = f._M[0] * G
    assert abs(Fz - Mg) / Mg * 100 < 1, f"Fz={Fz} Mg={Mg}"


@cuda_only
def test_pair_chunks_capture_equals_the_uncaptured_chunk_solve():
    """The captured chunk loop and the launch-per-colour chunk loop are the same solve, on the K=8 stack and
    on the 1e3 lattice."""
    def pair(centres, steps):
        eu = _engine(manifold_reduce=True, pair_chunks=True)
        eg = _engine(manifold_reduce=True, pair_chunks=True, graph_capture=True)
        for c in centres:
            eu.add_body(c)
            eg.add_body(c)
        for _ in range(steps):
            eu.step(1 / 240)
            eg.step(1 / 240)
            assert eu.n_pairs == eg.n_pairs and eu.n_colors == eg.n_colors
        assert float(np.max(np.abs(eu.get_state().xc - eg.get_state().xc))) == 0.0
        assert eg.n_graph_captures > 0

    pair([[0, 0, 0.101 + k * 0.205] for k in range(8)], 200)
    side = 23
    lat = [[ix * 0.35, iy * 0.35, 0.101 + lz * 0.205]
           for lz in range(2) for iy in range(side) for ix in range(side)][:1000]
    pair(lat, 12)


@cuda_only
def test_pair_chunks_require_the_device_manifold_stage():
    with pytest.raises(ValueError):
        _engine(pair_chunks=True)
    with pytest.raises(ValueError):
        _engine(manifold_reduce=True, manifold_on_host=True, pair_chunks=True)
    with pytest.raises(ValueError):
        _engine(manifold_reduce=True, pair_chunks=True, chunk_mode="wy")
