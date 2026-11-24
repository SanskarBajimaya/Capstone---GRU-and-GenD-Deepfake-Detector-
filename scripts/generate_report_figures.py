"""Rebuild the final capstone report figures with matplotlib (no proprietary data needed).

Data sources: saved result CSV files in the public capstone repository, commit
1489a95c4e6cb752874d7bdd8f59b041de333d13 (2026-11-19):
  temporal/results/validation_summary.csv
  temporal/results_pe/gru32_pe_validation_summary.csv
  temporal/results_pe/final_comparison_ffpp.csv
  temporal/results_pe/final_comparison_celebdf.csv
These compact CSV snapshots are supplied under data/ in the companion ZIP.
Run from the report package root: python scripts/generate_report_figures.py
All graphs and the two architecture schematics use matplotlib, and all text
inside figures uses at least 10 pt at the specified insertion size (3.26 in).
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
OUTPUT=ROOT/'latex_source'/'figures'
OUTPUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
 'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
 'xtick.labelsize':10,'ytick.labelsize':10, 'axes.spines.top':False,
 'axes.spines.right':False,'pdf.fonttype':42
})

def read(name):
 with (DATA/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def save(fig,n):
 fig.savefig(OUTPUT/n,dpi=300,bbox_inches='tight',pad_inches=.07,facecolor='white')
 plt.close(fig)

def nodes(ax,boxes,arrows):
 ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis('off')
 for key,(x,y,w,h,label) in boxes.items():
  rect=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12,rounding_size=.12',
      edgecolor='#235c76',linewidth=1.2,facecolor='#eef5f7')
  ax.add_patch(rect)
  ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10,weight='bold',linespacing=1.05)
 for src,dst in arrows:
  x,y,w,h,_=boxes[src]; x2,y2,w2,h2,_=boxes[dst]
  c1=(x+w/2,y+h/2);c2=(x2+w2/2,y2+h2/2)
  dx,dy=c2[0]-c1[0],c2[1]-c1[1]
  if abs(dx)>=abs(dy):a=(x+w if dx>0 else x,y+h/2);b=(x2 if dx>0 else x2+w2,y2+h2/2)
  else:a=(x+w/2,y+h if dy>0 else y);b=(x2+w2/2,y2 if dy>0 else y2+h2)
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,
                  color='#167d80',linewidth=1.25,shrinkA=2,shrinkB=2,
                  connectionstyle='arc3'))

fig,ax=plt.subplots(figsize=(3.26,2.34))
nodes(ax,{
 'a':(.3,4.58,2.5,1.0,'Video'), 'b':(3.80,4.58,2.5,1.0,'Frames'),
 'c':(7.17,4.58,2.5,1.0,'Face\nalign'),
 'd':(7.17,2.17,2.5,1.0,'Crop\n224 px'),
 'e':(3.80,2.17,2.5,1.0,'GenD\n1,024 D'),
 'f':(.3,2.17,2.5,1.0,'Mean\nscore')},
 [('a','b'),('b','c'),('c','d'),('d','e'),('e','f')])
save(fig,'fig01_gend_architecture.png')

fig,ax=plt.subplots(figsize=(3.26,2.44))
nodes(ax,{
 'a':(.3,4.7,2.55,.95,'Crops\n32'),
 'b':(3.80,4.7,2.55,.95,'CLIP\nfeat.'),
 'c':(7.17,4.7,2.55,.95,'CLIP\nGRU'),
 'd':(3.80,2.13,2.55,.95,'PE\nfeat.'),
 'e':(7.17,2.13,2.55,.95,'PE\nGRU'),
 'f':(3.80,.17,2.55,.85,'Score\navg.')},
 [('a','b'),('a','d'),('b','c'),('d','e'),('c','f'),('e','f')])
save(fig,'fig02_temporal_architecture.png')

# Six evaluation graphs. Names match the article's numbered captions.
NAMES={'Released GenD-CLIP (per-frame)':'CLIP head',
       'Released GenD-PE (per-frame)':'PE head',
       'Ensemble of released heads (CLIP+PE)':'Head avg.',
       'CLIP-GRU-32':'CLIP GRU', 'PE-GRU-32':'PE GRU',
       'Ensemble of GRUs (CLIP-GRU+PE-GRU)':'GRU avg.'}
rows_ff=read('final_comparison_ffpp.csv')
rows_cd=read('final_comparison_celebdf.csv')
rows_val=read('validation_summary.csv')
rows_pe=read('gru32_pe_validation_summary.csv')
colors=['#435e70','#435e70','#168c89','#1b728f','#1b728f','#c47f35']

def barh(n,labels,values,xlabel,xlims,fmt,colors=None):
 colors=colors or ['#52798e']*len(labels)
 # 3.26 inch exact target width keeps 10 point labels at print size.
 fig,ax=plt.subplots(figsize=(3.26,2.65))
 ax.barh(range(len(values)),values,color=colors,height=.68)
 ax.set_yticks(range(len(labels)),labels=labels)
 ax.invert_yaxis();ax.set_xlim(*xlims)
 ax.set_xlabel(xlabel,labelpad=3)
 ax.grid(axis='x',alpha=.17)
 ax.set_axisbelow(True)
 # Place values in the right margin when bars are near max; avoid overlapping.
 fig.subplots_adjust(left=.35,right=.95,top=.97,bottom=.25)
 for i,v in enumerate(values):
  ax.text(xlims[1]-(xlims[1]-xlims[0])*.01,i,fmt.format(v),
          fontsize=10,va='center',ha='right',color='#153343',weight='bold',
          bbox=dict(facecolor='white',alpha=.78,edgecolor='none',pad=.4))
 save(fig,n)

val=[float(next(r['auroc_mean'] for r in rows_val if r['model']==model and int(r['sequence_length'])==frames))
       for model,frames in [('GenD',32),('GRU',32),('GenD',64),('GRU',64)]]
val+= [float(next(r['auroc_mean'] for r in rows_pe if r['model']==model))
       for model in ['GenD-PE per-frame head','GRU-PE (mean of 5 seeds)']]
barh('fig03_validation_auroc.png',['CLIP head-32','CLIP GRU-32','CLIP head-64','CLIP GRU-64','PE head-32','PE GRU-32'],val,
     'Mean validation AUROC',(.995,1.001),'{:.4f}', ['#435e70','#1b728f','#435e70','#1b728f','#435e70','#1b728f'])

for fname,rows,key,xlabel,xlims,fmt in [
 ('fig04_ffpp_auroc.png',rows_ff,'auroc','FF++ test AUROC',(0,1.04),'{:.4f}'),
 ('fig06_celebdf_auroc.png',rows_cd,'auroc','Celeb-DF v2 AUROC',(0,1.04),'{:.4f}'),
 ('fig07_celebdf_ap.png',rows_cd,'ap','Celeb-DF v2 AP',(0,1.04),'{:.3f}'),
 ('fig08_celebdf_eer.png',rows_cd,'eer','Celeb-DF v2 EER (%)',(0,15.5),'{:.1f}'),
 ('fig09_celebdf_recall1.png',rows_cd,'recall_at_1pct_fpr','Recall at 1% FPR (%)',(0,70),'{:.1f}')]:
 labels=[NAMES[r['model']] for r in rows];values=[float(r[key])*(100 if key in ('eer','recall_at_1pct_fpr') else 1) for r in rows]
 barh(fname,labels,values,xlabel,xlims,fmt,colors)
# Supplementary evaluation plots required to preserve the accuracy comparison from
# slide 24 and to distinguish strict (1%) from moderate (5%) FPR operating points.
acc_labels=['CLIP head-32','CLIP GRU-32','CLIP head-64','CLIP GRU-64','PE head-32','PE GRU-32']
acc_values=[96.0,99.0,95.0,99.4,97.0,98.0]
barh('fig05_ffpp_accuracy.png',acc_labels,acc_values,'FF++ test accuracy (%)', (0,105),'{:.1f}',
     ['#435e70','#1b728f','#435e70','#1b728f','#435e70','#1b728f'])
barh('fig10_celebdf_recall5.png',[NAMES[r['model']] for r in rows_cd],
    [float(r['recall_at_5pct_fpr'])*100 for r in rows_cd],
    'Recall at 5% FPR (%)',(0,100),'{:.1f}',colors)
print('Wrote 10 report figures to',OUTPUT)
