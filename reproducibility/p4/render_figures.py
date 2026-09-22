#!/usr/bin/env python3
"""Render seven paper figures from only the named public CSV files."""
from __future__ import annotations
import csv
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent
DATA=HERE/"data"
OUT=HERE/"figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,
 "axes.labelsize":9,"xtick.labelsize":8.5,"ytick.labelsize":8.5,
 "legend.fontsize":8.5,"pdf.fonttype":42,"svg.fonttype":"none","svg.hashsalt":"P4-public-figures",
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
 "grid.color":"0.88","grid.linewidth":0.5,"savefig.dpi":220})
BLUE="#1f4e79"; GREEN="#24755c"; RED="#a63d40"; PURPLE="#70508c"; GREY="#777777"

def read(name):
    with (DATA/name).open(newline="") as f: return list(csv.DictReader(f))

def save(fig,stem):
    for ext in ("pdf","svg","png"):
        fig.savefig(OUT/f"{stem}.{ext}",bbox_inches="tight",pad_inches=.10)
    plt.close(fig)

def f1():
    d=read("f1.csv");x=np.arange(3)
    fig,ax=plt.subplots(1,2,figsize=(6.9,2.8),layout="constrained")
    v=[float(r["max_relative_residual"]) for r in d]
    ax[0].bar(x,v,color=BLUE,width=.52)
    ax[0].set_yscale("log");ax[0].set_ylim(1e-16,1e-11)
    ax[0].set_xticks(x,[r["scene"] for r in d]);ax[0].set_ylabel("Maximum relative residual")
    ax[0].set_title("(a) LP versus subset enumeration")
    for i,z in enumerate(v):ax[0].annotate(f"{z:.2g}",(i,z),xytext=(0,4),textcoords="offset points",ha="center",fontsize=8.5)
    p=[float(r["polygon_error_soc_pct"]) for r in d]
    ax[1].bar(x,p,color=GREEN,width=.52)
    ax[1].axhline(.9700556535,color=GREY,ls="--",lw=1.2,label="analytic maximum")
    ax[1].set_ylim(0,1.13);ax[1].set_xticks(x,[r["scene"] for r in d]);ax[1].set_ylabel("Maximum radial error (%)")
    ax[1].set_title("(b) 16-ray polygon versus cone")
    ax[1].legend(loc="lower right")
    save(fig,"f1_exactness")

def f2():
    d=read("f2.csv")
    fig,axs=plt.subplots(2,1,figsize=(6.9,4.5),layout="constrained",sharex=True)
    for ax,scene in zip(axs,("A","B")):
        a=[r for r in d if r["scene"]==scene]
        deg=np.array([float(r["direction_deg"]) for r in a]);lp=np.array([float(r["static_lp_m_s2"]) for r in a])
        lo=np.array([float(r["reached_low_m_s2"]) for r in a]);hi=np.array([float(r["reached_high_m_s2"]) for r in a])
        ax.plot(deg,lp,"o-",color=BLUE,label="static LP")
        ax.errorbar(deg,(lo+hi)/2,yerr=(hi-lo)/2,fmt="s--",color=RED,capsize=2,label="reached ramp bracket")
        ax.set_ylabel("Load (m/s²)");ax.set_title(f"({scene.lower()}) Anisotropic support {scene}; 12 sampled directions")
    axs[0].legend(loc="lower center",bbox_to_anchor=(.5,1.12),ncol=2)
    axs[1].set_xticks(np.arange(0,360,30));axs[1].set_xlabel("Load direction (degrees)")
    save(fig,"f2_reached")

def f3():
    d=read("f3.csv");full=d[:5];slow=d[5:]
    fig,ax=plt.subplots(figsize=(6.9,3.3),layout="constrained")
    ax.plot([float(r["duration_s"]) for r in full],[float(r["mean_abs_relative_error_pct"]) for r in full],"o-",color=RED,label="mean, 72 directions / 3 scenes")
    ax.plot([float(r["duration_s"]) for r in slow],[float(r["mean_abs_relative_error_pct"]) for r in slow],"s--",color=BLUE,label="one box direction")
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel("Ramp duration T (s; log scale)")
    ax.set_ylabel("Mean |ramp − static| / static (%)")
    ax.set_title("Ramp duration and dynamic excess load")
    ax.legend(loc="upper right");save(fig,"f3_ramp_rate")

def f4():
    d=read("f4.csv");x=np.array([float(r["direction_deg"]) for r in d]);lp=np.array([float(r["static_lp_m_s2"]) for r in d]);tip=np.array([float(r["tipping_m_s2"]) for r in d])
    fig,ax=plt.subplots(figsize=(6.9,3.4),layout="constrained")
    ax.plot(x,lp,"o--",color=BLUE,label="static wrench LP")
    ax.plot(x,tip,"x",color=RED,ms=8,mew=1.4,label="tipping limit; coincident")
    ax.set_xticks(x);ax.set_xlabel("Load direction (degrees)");ax.set_ylabel("Load (m/s²)")
    ax.set_title("Humanoid double support: static limit by direction")
    ax.legend(loc="upper left",ncol=1);ax.text(.02,.04,"Independently checked; limitations apply",transform=ax.transAxes,color=RED,fontsize=9,bbox=dict(facecolor="white",edgecolor="none",alpha=.9))
    save(fig,"f4_humanoid")

def f5():
    d=read("f5.csv");vals=np.array([float(r["static_radius_m_s2"]) for r in d]).reshape(21,21)
    miss=np.array([r["status"]!="feasible" for r in d]).reshape(21,21)
    fig,ax=plt.subplots(figsize=(6.9,4.3),layout="constrained")
    cmap=plt.get_cmap("viridis").copy();cmap.set_bad("#e6e6e6")
    im=ax.imshow(np.ma.masked_where(miss,vals),origin="lower",extent=(-.21,.21,-.21,.21),cmap=cmap,vmin=0,vmax=1.1,interpolation="nearest")
    ax.contourf(np.linspace(-.2,.2,21),np.linspace(-.2,.2,21),miss.astype(int),levels=[.5,1.5],colors="none",hatches=["////"])
    ax.set_xlabel("Moving-foot offset x (m)");ax.set_ylabel("Moving-foot offset y (m)")
    ax.set_title("Weakest sampled static radius across 12 directions")
    cb=fig.colorbar(im,ax=ax,shrink=.83);cb.set_label("Static radius (m/s²)")
    ax.text(.02,.97,"Independently checked; limitations apply",transform=ax.transAxes,va="top",color=RED,fontsize=9,bbox=dict(facecolor="white",edgecolor="none",alpha=.9))
    save(fig,"f5_placement")

def f6():
    d=read("f6.csv")
    fig=plt.figure(figsize=(6.9,4.4))
    ax=fig.add_axes([.30,.16,.68,.66])
    y=np.arange(len(d));safe=np.array([float(r["safe_cell_area"]) for r in d]);unsafe=np.array([float(r["unsafe_cell_area"]) for r in d]);unr=np.array([float(r["unresolved_cell_area"]) for r in d])
    ax.barh(y,safe,color=GREEN,edgecolor="black",linewidth=.3)
    ax.barh(y,unsafe,left=safe,color=RED,hatch="xxx",edgecolor="black",linewidth=.4)
    ax.barh(y,unr,left=safe+unsafe,color="#dedede",hatch="///",edgecolor="black",linewidth=.4)
    geometry={"g2":"Right foot +20°","g8":"Left foot +60°","g9":"Right foot −70°"}
    ax.set_yticks(y,[geometry[r["geometry"]]+"\n"+("reduced scalar" if r["method"]=="reduced scalar" else "full LP witness") for r in d]);ax.invert_yaxis()
    ax.set_xlim(0,450);ax.set_xlabel("Area (grid-cell equivalents; total 441)")
    fig.text(.64,.97,"Whole-cell areas at threshold 0.05 m/s²",ha="center",va="top",fontsize=10)
    fig.text(.64,.925,"Real arithmetic; outward rounding unverified",ha="center",va="top",fontsize=9,color=RED)
    fig.legend(handles=[Patch(facecolor=GREEN,edgecolor="black",label="safe by row method"),
                        Patch(facecolor=RED,edgecolor="black",hatch="xxx",label="unsafe"),
                        Patch(facecolor="#dedede",edgecolor="black",hatch="///",label="unresolved")],
               loc="upper center",bbox_to_anchor=(.64,.88),ncol=3,frameon=False)
    save(fig,"f6_regions")

def f7():
    d=read("f7_events.csv");m=read("f7_macro.csv")[0]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(6.9,4.0),layout="constrained",gridspec_kw={"width_ratios":[1.35,1]})
    y=np.arange(len(d));x=np.array([float(r["predicted_load"]) for r in d]);
    ax.scatter(x[:2],y[:2],marker="o",color=GREEN,label="predicted close")
    ax.scatter(x[2:],y[2:],marker="s",facecolors="none",edgecolors=RED,s=70,label="unconfirmed slip/graze")
    for i,r in enumerate(d[:2]):
        lo=float(r["branch_low"]);hi=float(r["branch_high"])
        ax.plot([lo,hi],[i+.10,i+.10],color="black",lw=2.2,zorder=4)
        ax.plot([lo,lo],[i+.07,i+.13],color="black",lw=1.5,zorder=4)
        ax.plot([hi,hi],[i+.07,i+.13],color="black",lw=1.5,zorder=4)
        ax.text(hi+.025,i+.10,f"[{lo:.3f}, {hi:.3f}]",va="center",fontsize=8.5)
    labels=["h=0.15 m, φ=0°","h=0.15 m, φ=270°","h=0.30 m, φ=15°","h=0.40 m, φ=15°"]
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,.8)
    ax.set_xlabel("Candidate local-event load (m/s²)");ax.set_title("(a) Branch confirmation")
    z=[float(m["first_local_load"]),float(m["collapse_load"]),float(m["macro_threshold_load"])]
    bx.bar(range(3),z,color=[BLUE,PURPLE,RED],width=.55)
    bx.set_xticks(range(3),["first\nslip","support\ncollapse","macro\nthreshold"])
    bx.set_ylabel("Dynamic-ramp load (m/s²)");bx.set_ylim(0,3.5)
    bx.set_title("(b) Same ramp, distinct events")
    bx.text(.02,.95,"A rate-set ramp; no branch-slip claim",transform=bx.transAxes,va="top",fontsize=8.5)
    save(fig,"f7_events")

if __name__=="__main__":
    for fn in (f1,f2,f3,f4,f5,f6,f7):fn()
