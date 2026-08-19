import struct, collections

TSIZE = {'b':1,'B':1,'c':1,'d':8,'f':4,'F':4,'G':16,'j':8,'J':8,'l':4,'L':4,
         'q':4,'Q':8,'s':2,'S':2,'U':16}
FMT = {'b':'b','B':'B','d':'d','f':'f','j':'q','J':'Q','l':'i','L':'I','s':'h','S':'H'}

def expand_type(t):
    out=[]; i=0
    while i < len(t):
        c=t[i]; i+=1
        n=1
        if i<len(t) and t[i]=='[':
            j=t.index(']',i); n=int(t[i+1:j]); i=j+1
        out.extend([c]*n)
    return out

def unpack(typechar, ssize, rpt, data, typestr):
    """returns list of tuples (one per sample)"""
    if typechar=='?' and typestr:
        types=expand_type(typestr)
        fmt='>'+''.join(FMT.get(c,'x') for c in types if c in FMT or c in ('F','c'))
        # build manually
        out=[]
        for r in range(rpt):
            chunk=data[r*ssize:(r+1)*ssize]; o=0; row=[]
            for c in types:
                if c=='F':
                    row.append(chunk[o:o+4].decode('latin1')); o+=4
                elif c=='c':
                    row.append(chunk[o:o+1].decode('latin1')); o+=1
                elif c in FMT:
                    sz=TSIZE[c]; row.append(struct.unpack('>'+FMT[c], chunk[o:o+sz])[0]); o+=sz
                else:
                    o+=TSIZE.get(c,1)
            out.append(tuple(row))
        return out
    if typechar=='F':
        return [(data[i*4:(i+1)*4].decode('latin1'),) for i in range(ssize*rpt//4)]
    if typechar=='c':
        return [(data[r*ssize:(r+1)*ssize].decode('latin1').strip('\x00'),) for r in range(rpt)]
    if typechar in FMT:
        n=ssize//TSIZE[typechar]
        vals=struct.unpack('>'+FMT[typechar]*(n*rpt), data[:TSIZE[typechar]*n*rpt])
        return [tuple(vals[r*n:(r+1)*n]) for r in range(rpt)]
    return []

def parse(path, want):
    """returns {key: [(payload_index, sample_tuple), ...]}, and header ctx"""
    buf=open(path,'rb').read()
    res=collections.defaultdict(list); ctx={}
    payload=[0]
    def walk(off,end,st):
        while off+8<=end:
            key=buf[off:off+4].decode('latin1')
            t=chr(buf[off+4]) if buf[off+4] else '\0'
            ss=buf[off+5]; rpt=struct.unpack('>H',buf[off+6:off+8])[0]
            dl=ss*rpt; data=buf[off+8:off+8+dl]
            if t=='\0':
                walk(off+8, off+8+dl, dict(st) if key=='STRM' else st)
            else:
                if key=='SCAL':
                    st['SCAL']=[v[0] for v in unpack(t,ss,rpt,data,None)] if ss==TSIZE.get(t,1) else list(unpack(t,ss,rpt,data,None)[0])
                elif key=='TYPE':
                    st['TYPE']=data.decode('latin1').strip('\x00')
                elif key in want:
                    rows=unpack(t,ss,rpt,data,st.get('TYPE'))
                    scal=st.get('SCAL')
                    if scal and key not in ('TMPC',):
                        rows=[tuple(v/ (scal[i] if i<len(scal) else scal[-1]) if isinstance(v,(int,float)) else v
                                    for i,v in enumerate(row)) for row in rows]
                    res[key].append((payload[0], rows))
                elif t=='c' and key not in ('STNM','RMRK'):
                    ctx.setdefault(key, data.decode('latin1').strip('\x00').strip())
            off+=8+dl+((-dl)%4)
    off=0
    while off+8<=len(buf):
        if buf[off:off+4]!=b'DEVC': off+=4; continue
        dl=buf[off+5]*struct.unpack('>H',buf[off+6:off+8])[0]
        walk(off+8, min(off+8+dl,len(buf)), {})
        payload[0]+=1
        off+=8+dl+((-dl)%4)
    return res, ctx, payload[0]
