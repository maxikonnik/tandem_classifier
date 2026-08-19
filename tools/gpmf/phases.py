import sys,os,json
sys.path.insert(0,'/tmp/an')
import numpy as np
from analyze import analyze

def segment(T):
    n=len(T)
    ev=np.array([1/r['shut'] if r['shut'] else 0 for r in T])   # 1/выдержка
    iso=np.array([r['iso'] for r in T])
    indo=np.array([r['INDO'] for r in T])
    gmax=np.array([r['g_max'] for r in T])
    gmean=np.array([r['g_mean'] for r in T])
    # ВЫХОД: первый устойчивый переход к яркому свету (ISO=100 и очень короткая выдержка)
    bright=(iso<=110)&(ev>1800)
    exit_t=None
    for i in range(n-3):
        if bright[i:i+4].all(): exit_t=i; break
    # РАСКРЫТИЕ: макс. перегрузка после выхода (+8 c, чтобы не поймать отделение)
    open_t=None
    if exit_t is not None:
        s=exit_t+8
        if s<n:
            score=gmax[s:]*0.6+gmean[s:]*1.4
            open_t=s+int(np.argmax(score))
    return exit_t,open_t

def phases(T):
    e,o=segment(T); n=len(T)
    lab=['?']*n
    for i in range(n):
        if e is None: lab[i]='?'
        elif i<e: lab[i]='самолёт'
        elif o is not None and i<o-1: lab[i]='свободное падение'
        elif o is not None and abs(i-o)<=1: lab[i]='раскрытие'
        else: lab[i]='под куполом'
    return lab,e,o

FIELDS=['g_mean','g_max','g_std','gyro_rms','faces','face_conf','smile','face_area','yavg','unif','iso','tmpc','SNOW','URBA','INDO','WATR','VEGE','BEAC']
def report(path):
    T,ctx,n=analyze(path,'')
    for r in T: r['ev']=1/r['shut'] if r['shut'] else 0
    lab,e,o=phases(T)
    order=['самолёт','свободное падение','раскрытие','под куполом']
    print(f"\n### {os.path.basename(path)}  ({n} с, {ctx.get('DVNM')})")
    print(f"    выход ≈ {e} с, раскрытие ≈ {o} с, свободное падение ≈ {o-e if e is not None and o else '?'} с")
    print(f"    {'параметр':<12}"+''.join(f"{p:>20}" for p in order))
    for f in FIELDS+['ev']:
        line=f"    {f:<12}"
        for p in order:
            v=[T[i][f] for i in range(n) if lab[i]==p]
            line+=f"{np.mean(v):>20.2f}" if v else f"{'—':>20}"
        print(line)
    return T,lab,e,o,ctx

if __name__=='__main__':
    report(sys.argv[1])
