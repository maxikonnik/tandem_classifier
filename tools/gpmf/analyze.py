import sys,os,subprocess,json,math
sys.path.insert(0,'/tmp/an')
import numpy as np
from gpmf_full import parse

WANT={'GPS9','SCEN','FACE','ACCL','GYRO','GRAV','CORI','SHUT','ISOE','TMPC','YAVG','UNIF','HUES','WBAL'}

def extract(path,binout='/tmp/an/x.bin'):
    pr=subprocess.run(['ffprobe','-v','error','-show_entries','stream=index,codec_tag_string','-of','json',path],capture_output=True,text=True)
    idx=[s['index'] for s in json.loads(pr.stdout)['streams'] if s.get('codec_tag_string')=='gpmd'][0]
    subprocess.run(['ffmpeg','-v','error','-y','-i',path,'-map',f'0:{idx}','-c','copy','-f','data',binout],capture_output=True)
    return parse(binout,WANT)

def per_second(res,key,n):
    """group samples by payload index (~1s)"""
    out=[[] for _ in range(n)]
    for pl,rows in res[key]:
        if pl<n: out[pl].extend(rows)
    return out

def analyze(path,label):
    res,ctx,n=extract(path)
    acc=per_second(res,'ACCL',n); gyr=per_second(res,'GYRO',n)
    scen=per_second(res,'SCEN',n); face=per_second(res,'FACE',n)
    hues=per_second(res,'HUES',n); yavg=per_second(res,'YAVG',n)
    shut=per_second(res,'SHUT',n); iso=per_second(res,'ISOE',n)
    tmpc=per_second(res,'TMPC',n); unif=per_second(res,'UNIF',n)
    T=[]
    for i in range(n):
        a=np.array(acc[i]) if acc[i] else np.zeros((1,3))
        g=np.array(gyr[i]) if gyr[i] else np.zeros((1,3))
        amag=np.linalg.norm(a,axis=1)/9.80665
        gmag=np.linalg.norm(g,axis=1)
        sc={}
        for row in scen[i]: sc.setdefault(row[0].strip(),[]).append(row[1])
        sc={k:float(np.mean(v)) for k,v in sc.items()}
        fc=[r for r in face[i] if r[1]>=50]        # confidence >= 50%
        hue=None
        if hues[i]:
            h=np.array(hues[i],dtype=float); w=h[:,1]
            hue=float(h[np.argmax(w),0])*360/255
        T.append(dict(t=i,
            g_mean=float(amag.mean()), g_max=float(amag.max()), g_std=float(amag.std()),
            gyro_rms=float(np.sqrt((gmag**2).mean())), gyro_max=float(gmag.max()),
            faces=len(fc), face_conf=float(np.mean([r[1] for r in fc])) if fc else 0,
            smile=float(np.mean([r[7] for r in fc])) if fc else 0,
            blink=float(np.mean([r[8] for r in fc])) if fc else 0,
            face_area=float(np.mean([r[5]*r[6] for r in fc])) if fc else 0,
            hue=hue, yavg=float(np.mean(yavg[i])) if yavg[i] else 0,
            unif=float(np.mean(unif[i])) if unif[i] else 0,
            shut=float(np.mean(shut[i])) if shut[i] else 0,
            iso=float(np.mean(iso[i])) if iso[i] else 0,
            tmpc=float(np.mean(tmpc[i])) if tmpc[i] else 0,
            **{k:sc.get(k,0) for k in ('SNOW','URBA','INDO','WATR','VEGE','BEAC')}))
    return T,ctx,n

if __name__=='__main__':
    p=sys.argv[1]
    T,ctx,n=analyze(p,'x')
    hdr=f"{'t':>4} {'g_avg':>6} {'g_max':>6} {'g_sd':>5} {'gyroRMS':>7} {'лиц':>4} {'conf':>4} {'smile':>5} {'hue':>5} {'YAVG':>5} {'ISO':>5} {'1/shut':>7} {'T°C':>5} | SNOW URBA INDO WATR VEGE BEAC"
    print(f"камера {ctx.get('DVNM')}, {n} с\n"+hdr)
    for r in T:
        print(f"{r['t']:4d} {r['g_mean']:6.2f} {r['g_max']:6.2f} {r['g_std']:5.2f} {r['gyro_rms']:7.2f} "
              f"{r['faces']:4d} {r['face_conf']:4.0f} {r['smile']:5.0f} "
              f"{(r['hue'] or 0):5.0f} {r['yavg']:5.0f} {r['iso']:5.0f} {1/r['shut'] if r['shut'] else 0:7.0f} {r['tmpc']:5.1f} | "
              +' '.join(f"{r[k]:4.2f}" for k in ('SNOW','URBA','INDO','WATR','VEGE','BEAC')))
