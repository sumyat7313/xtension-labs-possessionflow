"""Render the frozen final-test comparison; requires matplotlib for this optional plot."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary",type=Path)
    parser.add_argument("--out",type=Path,default=Path("assets/control-benchmark.png"))
    args=parser.parse_args()
    summary=json.loads(args.summary.read_text(encoding="utf-8"))
    winner=summary["selected_on_validation"]
    rows=summary["final_records"]
    if not rows: raise ValueError("No final results: no candidate was eligible for final testing")
    cells=sorted({(r["match"],r["regime"]) for r in rows})
    lookup={(r["match"],r["regime"],r["candidate"]):r for r in rows}
    baseline=np.array([lookup[m,r,"release-1.1"]["spellF1"] for m,r in cells])
    selected=np.array([lookup[m,r,winner]["spellF1"] for m,r in cells])
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})
    bg,fg,muted,mint="#07151c","#f3f5f3","#9baeb6","#64e3b5"
    fig,ax=plt.subplots(figsize=(11.5,6.2),facecolor=bg)
    ax.set_facecolor(bg)
    y=np.arange(len(cells))
    for i,(before,after) in enumerate(zip(baseline,selected)):
        ax.plot([before,after],[i,i],color=muted,linewidth=2,zorder=2)
        ax.text(max(before,after)+.012,i,f"{after-before:+.3f}",va="center",color=mint if after>=before else "#ff9c87",fontsize=11)
    ax.scatter(baseline,y,s=75,color=muted,label="Released v1.1",zorder=3)
    ax.scatter(selected,y,s=100,color=mint,label="Frozen v2 candidate",zorder=4)
    ax.set_yticks(y,[f"{match}  /  {regime}" for match,regime in cells],color=fg)
    ax.invert_yaxis()
    ax.set_xlim(0,max(.6,float(max(selected.max(),baseline.max()))+.12))
    ax.set_xlabel("Possession-spell F1  ·  official temporal IoU metric",color=fg,labelpad=14)
    ax.tick_params(axis="x",colors=muted)
    ax.tick_params(axis="y",length=0,pad=15)
    ax.grid(axis="x",color="#20323c",alpha=.8)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_axisbelow(True)
    legend=ax.legend(loc="lower right",facecolor=bg,edgecolor="#20323c",labelcolor=fg)
    fig.text(.07,.94,"XTENSION LABS  /  POSSESSIONFLOW",color=mint,fontsize=11,fontweight="bold")
    fig.text(.07,.875,"Three fresh matches. One frozen comparison.",color=fg,fontsize=21,fontweight="bold")
    fig.text(.07,.815,f"Equal-cell mean: {baseline.mean():.3f} → {selected.mean():.3f}  |  Δ {np.mean(selected-baseline):+.3f}",color=muted,fontsize=13)
    fig.text(.07,.035,"External Sportec matches; clean/degraded inputs share each match. Not a competition leaderboard score.",color=muted,fontsize=9)
    fig.subplots_adjust(left=.24,right=.94,top=.73,bottom=.16)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out,dpi=180,facecolor=bg)
    print(args.out.resolve())
if __name__=="__main__":main()
