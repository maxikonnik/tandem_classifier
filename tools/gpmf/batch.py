import sys,os,csv,warnings
sys.path.insert(0,'/tmp/an'); warnings.filterwarnings('ignore')
import numpy as np
from analyze import analyze
from phases import phases, FIELDS
root="/sessions/charming-intelligent-ride/mnt/Claude/tandem_detector/Samples"
rows=list(csv.DictReader(open('/sessions/charming-intelligent-ride/mnt/Claude/tandem_detector/samples_telemetry_inventory.csv',encoding='utf-8-sig')))
rows=[r for r in rows if r['Камера']=='HERO11 Black' and float(r['Длит., с'])>80]
rows.sort(key=lambda r:-float(r['Длит., с']))
agg={p:{f:[] for f in FIELDS+['ev']} for p in ['самолёт','свободное падение','раскрытие','под куполом']}
summ=[]
for r in rows:
    f=os.path.join(root,r['Папка'],r['Файл'])
    try:
        T,ctx,n=analyze(f,'')
    except Exception as ex:
        print("ERR",r['Файл'],ex); continue
    for x in T: x['ev']=1/x['shut'] if x['shut'] else 0
    lab,e,o=phases(T)
    if e is None or o is None: summ.append((r['Папка'],r['Файл'],n,None,None)); continue
    summ.append((r['Папка'],r['Файл'],n,e,o))
    for p in agg:
        for fl in agg[p]:
            v=[T[i][fl] for i in range(n) if lab[i]==p]
            if v: agg[p][fl].append(np.mean(v))
print(f"{'сек':>5} {'выход':>6} {'раскр':>6} {'СП,с':>5}  файл")
for pa,fn,n,e,o in summ:
    print(f"{n:5d} {str(e):>6} {str(o):>6} {str(o-e) if e is not None and o else '—':>5}  {pa}/{fn}")
import pickle; pickle.dump(agg,open('/tmp/an/agg.pkl','wb'))
print("\n=== СРЕДНЕЕ ПО",len([s for s in summ if s[3] is not None]),"ПРЫЖКАМ ===")
order=['самолёт','свободное падение','раскрытие','под куполом']
print(f"{'параметр':<12}"+''.join(f"{p:>19}" for p in order))
for fl in FIELDS+['ev']:
    line=f"{fl:<12}"
    for p in order:
        v=agg[p][fl]
        line+=f"{np.mean(v):>13.2f}±{np.std(v):>5.2f}" if v else f"{'—':>19}"
    print(line)
