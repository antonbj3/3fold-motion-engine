#!/usr/bin/env python3
"""
cell 529 — the V6 TERMINAL made quantitative: the σ_min-IDENTIFIABILITY CLIMB of the contact-geometry parameters (κ1,κ2) as you read more of the 2-jet, MEASURED
at the REAL SO-ARM100 curvatures from my cell 528 backend. The curated V6 row names the terminal = "I's σ_min-vector identifiability↔gauge boundary made physical",
and the ladder = {penetration·normal·mean·anisotropy·orientation}. cell 196 argued the observable-rank hierarchy QUALITATIVELY (force→+mean→+full-spectrum); cell 528
showed the tropical-SDF backend SUPPLIES those rungs on a real robot. cell 529 MEASURES the rank climb with MY σ_min machinery (cert-spine): the identifiability of
the two principal curvatures (κ1,κ2) = σ_min of the observable Jacobian ∂(observables)/∂(κ1,κ2), row-normalized so σ_min∈[0,1] is the ANGULAR independence of
the observable gradients ([[feature-fit-identifiability-angular-not-fraction]]: identifiability is ANGULAR). The physics of each rung: FORCE (elliptical
Hertz) depends on (κ1,κ2) ONLY through the effective curvature κ_eff=√(κ1κ2) (a single symmetric combination) → its gradient is rank-1 in (κ1,κ2) → the
ANTI-DIAGONAL (κ1↑κ2↓ at fixed κ_eff) is a NULL direction → σ_min=0 (C-geo↔C-man COUPLED, I's V6 verdict). Adding the 2-jet MEAN curvature (Hessian trace
κ1+κ2 = a DIFFERENT symmetric combination) makes {κ_eff, mean} = the two elementary symmetric polynomials → identifiable, BUT σ_min→0 as κ1→κ2 (at ISOTROPY the
two curvatures are genuinely indistinguishable — an intrinsic degeneracy, not a cert failure). The FULL SPECTRUM (κ1,κ2 directly) → σ_min high, over-determined,
robust EVEN at isotropy. PREREGISTERED (watertight, at the real SO-ARM100 curvatures): (a) force-only σ_min≈0 (rank-1, anti-diagonal null); (b) +mean σ_min JUMPS at the
ANISOTROPIC cylinder contact (κ1≠κ2) but stays ≈0 at the ISOTROPIC cap (κ1≈κ2, intrinsic); (c) +full-spectrum σ_min high at BOTH (over-determined). ⟹ the σ_min
climb IS the V6 observable-rank ladder, and identifiability itself is anisotropy-gated. Anchors: cell 528 (real curvatures), cell 196 (rung hierarchy), cell 195 (2-jet
decouple), fisher-lambda-min (σ_min=identifiability), feature-fit-identifiability-angular, adjudicate-by-overdetermination, S2 σ_min-crown. External: SO-ARM100.
"""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(_v,"4")  # thread cap (thread cap)
import numpy as np, importlib.util
def _load(n,p):
    p=p if os.path.isabs(p) else os.path.join(os.path.dirname(os.path.abspath(__file__)),p)
    p=p if os.path.isabs(p) else os.path.join(os.path.dirname(os.path.abspath(__file__)),p)
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
w528=_load("cell 528", "tropical_sdf_backend_computes_full_v6_contact_geometry_ladder_2jet_curvature_anisotropy_on_so_arm100.py")
w527=w528.w527; posed_capsules=w527.posed_capsules

def observables(k1,k2,rung):
    """observable vector as a function of the two principal curvatures (κ1,κ2), per ladder rung."""
    keff=np.sqrt(max(k1*k2,1e-12))                                   # FORCE sees only κ_eff=√(κ1κ2) (elliptical Hertz effective curvature)
    if rung=="force":    return np.array([keff])
    if rung=="+mean":    return np.array([keff, k1+k2])              # + 2-jet Hessian TRACE (mean curvature)
    if rung=="+spectrum":return np.array([keff, k1+k2, k1, k2])     # + full 2-jet SPECTRUM (both eigenvalues)

def sigma_min_ident(k1,k2,rung,h=1e-3):
    """σ_min of the ROW-NORMALIZED observable Jacobian ∂obs/∂(κ1,κ2) at (κ1,κ2) → angular identifiability ∈[0,1] (0=rank-deficient/null direction)."""
    o1p=observables(k1+h,k2,rung); o1m=observables(k1-h,k2,rung); o2p=observables(k1,k2+h,rung); o2m=observables(k1,k2-h,rung)
    J=np.stack([(o1p-o1m)/(2*h),(o2p-o2m)/(2*h)],axis=1)            # (n_obs × 2) Jacobian [∂/∂κ1, ∂/∂κ2]
    rn=np.linalg.norm(J,axis=1,keepdims=True); J=J/np.maximum(rn,1e-12)   # row-normalize → angular independence
    s=np.linalg.svd(J,compute_uv=False); return float(s[-1]) if J.shape[0]>=2 else 0.0   # σ_min (rank-1 single row → 0 in 2 params)

def real_curvatures():
    """the REAL SO-ARM100 contact curvatures from the cell 528 backend (cylinder = anisotropic, EE-tip cap = isotropic)."""
    caps=posed_capsules(np.zeros(5)); a,b,r=caps[2]; ax=(b-a)/np.linalg.norm(b-a)
    perp=np.cross(ax,[0,1,0]); perp=perp/np.linalg.norm(perp)
    p_cyl=0.5*(a+b)+r*perp; aL,bL,rL=caps[-1]; axL=(bL-aL)/np.linalg.norm(bL-aL); p_cap=bL+rL*axL
    kc=w528.curvatures(p_cyl,caps); ks=w528.curvatures(p_cap,caps)
    return {"CYLINDER (anisotropic)":(abs(kc[0]),abs(kc[1])), "EE-CAP (isotropic)":(abs(ks[0]),abs(ks[1]))}

def main():
    print("="*124); print("cell 529  V6 TERMINAL: σ_min-IDENTIFIABILITY CLIMB of contact curvature (force→+mean→+spectrum) at REAL SO-ARM100 curvatures (cell 528⊗σ_min)"); print("="*124)
    curv=real_curvatures()
    print("\n  REAL SO-ARM100 contact curvatures (from the cell 528 tropical-SDF backend):")
    for name,(k1,k2) in curv.items(): print("    %-26s (κ1,κ2)=(%.2f, %.2f)  anisotropy|κ1−κ2|=%.2f" % (name,k1,k2,abs(k1-k2)))

    rungs=["force","+mean","+spectrum"]
    print("\n  σ_min-IDENTIFIABILITY of (κ1,κ2) per observable rung (0=rank-deficient/null direction, 1=orthogonal/robust):")
    print("    %-26s %-12s %-12s %-12s" % ("contact","force","+mean","+spectrum"))
    S={}
    for name,(k1,k2) in curv.items():
        row=[sigma_min_ident(k1,k2,rg) for rg in rungs]; S[name]=row
        print("    %-26s %-12.3f %-12.3f %-12.3f" % (name,row[0],row[1],row[2]))

    # anisotropy SWEEP (robustness — don't over-rely on the κ2=0 cylinder endpoint where the elliptical-Hertz κ_eff degenerates to a LINE contact):
    # +mean identifiability as κ2/κ1 goes isotropic(1)→anisotropic, at κ1 fixed to the cap scale. Should climb SMOOTHLY from 0 → high.
    k1f=30.0; print("\n  ANISOTROPY-GATING (sweep κ2/κ1 at κ1=%.0f; +mean rung) — σ_min climbs smoothly with anisotropy, robust to the κ2→0 line-contact edge:" % k1f)
    print("    κ2/κ1     %s" % "  ".join("%.2f"%ratio for ratio in [1.0,0.7,0.4,0.2,0.05]))
    print("    σ_min     %s" % "  ".join("%.3f"%sigma_min_ident(k1f,ratio*k1f,"+mean") for ratio in [1.0,0.7,0.4,0.2,0.05]))
    sweep=[sigma_min_ident(k1f,ratio*k1f,"+mean") for ratio in [1.0,0.7,0.4,0.2,0.05]]
    monotone = all(sweep[i]<=sweep[i+1]+1e-6 for i in range(len(sweep)-1)) and sweep[0]<0.1 and sweep[-1]>0.3   # 0 at isotropy → high at anisotropy

    an="CYLINDER (anisotropic)"; iso="EE-CAP (isotropic)"
    force_null   = S[an][0]<0.05 and S[iso][0]<0.05                                       # (a) force alone: rank-1, σ_min≈0 both
    mean_climbs  = S[an][1]>0.2                                                           # (b) +mean identifies the ANISOTROPIC contact
    iso_degenerate = S[iso][1]<0.1                                                        # (b) +mean still ≈0 at ISOTROPY (intrinsic κ1=κ2 degeneracy)
    spectrum_robust = S[an][2]>0.3 and S[iso][2]>0.3                                       # (c) +full spectrum robust at BOTH
    print("\n  ANALYSIS: force-null-both=%s | +mean-climbs-at-anisotropic=%s | +mean-still-degenerate-at-ISOTROPY(intrinsic)=%s | +spectrum-robust-BOTH=%s | anisotropy-sweep-monotone=%s" %
          (force_null,mean_climbs,iso_degenerate,spectrum_robust,monotone))

    print("\n  VERDICT (does the σ_min-identifiability climb force→+mean→+spectrum, at the real SO-ARM100 curvatures — the V6 terminal quantified?):")
    if force_null and mean_climbs and iso_degenerate and spectrum_robust and monotone:
        print("  OK:  DELIVERED (the V6 TERMINAL made quantitative on a real robot — my σ_min identifiability machinery ⊗ cell 528's curvature ladder ⊗ cell 196's rung")
        print("    hierarchy) — the identifiability of the contact geometry (κ1,κ2) = σ_min of the observable Jacobian CLIMBS as you read more of the 2-jet, at")
        print("    the REAL SO-ARM100 curvatures: (a) FORCE alone → σ_min≈0 at BOTH contacts (force sees only κ_eff=√(κ1κ2), a single symmetric combination →")
        print("    the anti-diagonal κ1↑κ2↓ is a NULL direction → rank-1 → C-geo↔C-man COUPLED, I's V6 verdict quantified); (b) +2-jet MEAN curvature (Hessian")
        print("    trace) → σ_min JUMPS to %.2f at the ANISOTROPIC cylinder (κ1≠κ2 → {κ_eff,mean} are 2 independent symmetric polys → identifiable), but stays" % S[an][1])
        print("    ≈%.2f at the ISOTROPIC cap — an INTRINSIC degeneracy (κ1=κ2 are genuinely indistinguishable, NOT a cert failure); (c) +FULL SPECTRUM → σ_min" % S[iso][1])
        print("    robust (%.2f/%.2f) at BOTH (over-determined, reads κ1,κ2 directly). => THE σ_min CLIMB IS THE V6 OBSERVABLE-RANK LADDER, and identifiability" % (S[an][2],S[iso][2]))
        print("    is ANISOTROPY-GATED: +mean suffices to identify an anisotropic contact but the full 2-jet spectrum is needed for robustness + at isotropy.")
        print("    This turns cell 196's qualitative hierarchy into a measured σ_min curve on the SO-ARM100 backend (cell 528), grounding the V6 terminal that @I's")
        print("    σ_min-vector defines — decorrelated legs: MY σ_min-of-observable-Jacobian ⊗ @I's σ_min-identifiability method (adjudicate-by-over-det). =>")
        print("    for the contact cert: report WHICH 2-jet rungs are read → the σ_min tells you if the contact geometry is identifiable (and at isotropy, that")
        print("    κ1,κ2 SHOULD collapse). HONEST SCOPE: the SO-ARM100 cylinder has κ2=0 EXACTLY (line contact) where the elliptical-Hertz κ_eff degenerates —")
        print("    so I did NOT over-rely on that endpoint: the anisotropy SWEEP (κ2/κ1 1→0.05) shows σ_min(+mean) climbs SMOOTHLY %s (monotone, 0 at isotropy →" % [round(x,2) for x in sweep])
        print("    high at anisotropy), the real-robot points are two samples on this curve. σ: force σ_min≈0 both; +mean %.2f(aniso)/%.2f(iso) + monotone sweep;" % (S[an][1],S[iso][1]))
        print("    +spectrum %.2f/%.2f; anisotropy-gated identifiability." % (S[an][2],S[iso][2]))
    else:
        print("  ◐ force_null=%s mean_climbs=%s iso_degenerate=%s spectrum_robust=%s (S=%s) — inspect." % (force_null,mean_climbs,iso_degenerate,spectrum_robust,{k:[round(x,2) for x in v] for k,v in S.items()}))

if __name__=="__main__": main()
