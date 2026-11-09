"""Phase 0: group timestamped source photos into capture bursts.

Run from the repository root:  python synthetic-study/phase0/capture_groups.py
Read-only; prints counts used in UNDERSTANDING.md section 4.
"""
import os,re,json,sys,collections,datetime as dt
R=re.compile(r'^(.+)_(\d+)_(\d+)_(\d+)_(\d+)\.png$')
tiles=[]
for s in ['train','val','test']:
  for c in ['Healthy','Diseased']:
    for f in os.listdir(f'DatasetPreped2/images/{s}/{c}'):
      tiles.append((s,c,R.match(f).group(1)))
srcs=sorted({t[2] for t in tiles})
ts={}
for k in srcs:
  m=re.match(r'img_(\d{8})_(\d{6})',k)
  if m: ts[k]=dt.datetime.strptime(m.group(1)+m.group(2),'%Y%m%d%H%M%S')
print('sources',len(srcs),'timestamped',len(ts))
hs={t[2] for t in tiles if t[1]=='Healthy'}
print('healthy tiles',sum(t[1]=='Healthy' for t in tiles),'healthy sources',len(hs),'healthy timestamped',len(hs&set(ts)))
for gap in [10,30,60,300]:
  order=sorted(ts,key=ts.get); g={}; gid=0; prev=None
  for k in order:
    if prev is None or (ts[k]-ts[prev]).total_seconds()>gap: gid+=1
    g[k]=gid; prev=k
  hg={g[k] for k in hs if k in g}
  # how many val/test tiles share a time-group with a train tile
  tr={g[t[2]] for t in tiles if t[0]=='train' and t[2] in g}
  sh=sum(1 for t in tiles if t[0]!='train' and t[2] in g and g[t[2]] in tr)
  tot=sum(1 for t in tiles if t[0]!='train' and t[2] in g)
  print(f'gap<= {gap}s: {gid} time-groups from {len(ts)} timestamped photos; healthy timestamped photos fall in {len(hg)} groups; val/test timestamped tiles sharing a group with train: {sh}/{tot}')
