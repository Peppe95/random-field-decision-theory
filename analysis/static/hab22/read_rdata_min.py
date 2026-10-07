import gzip, struct, math, sys

NILVALUE_SXP=254; REFSXP=255; SYMSXP=1; LISTSXP=2; LANGSXP=6; CHARSXP=9; LGLSXP=10; INTSXP=13; REALSXP=14; CPLXSXP=15; STRSXP=16; VECSXP=19; EXPRSXP=20; RAWSXP=24; ALTREP_SXP=238
ATTR_BIT=1<<9; TAG_BIT=1<<10; OBJECT_BIT=1<<8

class RObj:
    __slots__=('type','data','attrs','tag','object','flags')
    def __init__(self,t,data=None,attrs=None,tag=None,object=False,flags=0):
        self.type=t; self.data=data; self.attrs=attrs; self.tag=tag; self.object=object; self.flags=flags
    def __repr__(self): return f'RObj(type={self.type}, data={type(self.data).__name__}, attrs={self.attrs is not None})'

class Reader:
    def __init__(self,f):
        self.f=f; self.refs=[]; self.counts={}; self.depth=0
    def pos(self): return self.f.tell()
    def readn(self,n):
        b=self.f.read(n)
        if len(b)!=n: raise EOFError((n,len(b),self.pos()))
        return b
    def i32(self): return struct.unpack('>i',self.readn(4))[0]
    def u32(self): return struct.unpack('>I',self.readn(4))[0]
    def f64s(self,n):
        b=self.readn(8*n)
        return list(struct.unpack('>'+('d'*n),b))
    def addref(self,obj):
        self.refs.append(obj); return obj
    def item(self):
        pos=self.pos(); flags=self.u32(); t=flags & 0xff
        self.counts[t]=self.counts.get(t,0)+1
        if t==REFSXP:
            idx=flags>>8
            if idx==0: idx=self.i32()
            if idx<1 or idx>len(self.refs):
                raise ValueError(f'bad ref {idx} at {pos}, refs={len(self.refs)}, flags={flags:#x}')
            return self.refs[idx-1]
        if t==NILVALUE_SXP:
            return None
        if t in (251,252,253,250,241,242):
            return RObj(t)
        has_attr=bool(flags & ATTR_BIT); has_tag=bool(flags & TAG_BIT); objbit=bool(flags & OBJECT_BIT)
        # pairlists/language: attrs/tag are serialized before car/cdr
        if t in (LISTSXP, LANGSXP, 17):
            ro=RObj(t, flags=flags, object=objbit); self.addref(ro)
            attrs=self.item() if has_attr else None
            tag=self.item() if has_tag else None
            car=self.item(); cdr=self.item()
            ro.data=(car,cdr); ro.attrs=attrs; ro.tag=tag
            return ro
        if t==SYMSXP:
            ro=RObj(t, flags=flags); self.addref(ro)
            ro.data=self.item()
            return ro
        if t==CHARSXP:
            n=self.i32()
            if n==-1: s=None
            else: s=self.readn(n).decode('utf-8','replace')
            ro=RObj(t,s,flags=flags)
            # CHARSXP refs? add to be safe
            self.addref(ro)
            return ro
        if t in (LGLSXP, INTSXP):
            n=self.i32(); vals=[self.i32() for _ in range(n)]
            ro=RObj(t,vals,flags=flags,object=objbit)
            if has_attr: ro.attrs=self.item()
            return ro
        if t==REALSXP:
            n=self.i32(); vals=self.f64s(n)
            ro=RObj(t,vals,flags=flags,object=objbit)
            if has_attr: ro.attrs=self.item()
            return ro
        if t==CPLXSXP:
            n=self.i32(); vals=[]
            for _ in range(n): vals.append(complex(*struct.unpack('>dd',self.readn(16))))
            ro=RObj(t,vals,flags=flags,object=objbit)
            if has_attr: ro.attrs=self.item()
            return ro
        if t==STRSXP:
            n=self.i32(); ro=RObj(t,[],flags=flags,object=objbit)
            # regular vectors aren't refs normally; don't add
            ro.data=[self.item() for _ in range(n)]
            if has_attr: ro.attrs=self.item()
            return ro
        if t in (VECSXP,EXPRSXP):
            n=self.i32(); ro=RObj(t,[],flags=flags,object=objbit)
            ro.data=[self.item() for _ in range(n)]
            if has_attr: ro.attrs=self.item()
            return ro
        if t==RAWSXP:
            n=self.i32(); ro=RObj(t,self.readn(n),flags=flags,object=objbit)
            if has_attr: ro.attrs=self.item()
            return ro
        if t==ALTREP_SXP:
            # R serialization: info, state, attr
            ro=RObj(t,flags=flags,object=objbit); self.addref(ro)
            ro.data=(self.item(),self.item())
            ro.attrs=self.item()
            return ro
        raise NotImplementedError(f'type {t} flags={flags:#x} pos={pos} counts={self.counts}')

def unwrap_char(x):
    if x is None: return None
    if isinstance(x,RObj) and x.type==CHARSXP: return x.data
    if isinstance(x,RObj) and x.type==SYMSXP: return unwrap_char(x.data)
    return x

def attrs_to_dict(a):
    out={}
    while isinstance(a,RObj) and a.type==LISTSXP:
        car,cdr=a.data
        out[unwrap_char(a.tag)] = car
        a=cdr
    return out

def vec_values(o):
    if o is None: return None
    if o.type==STRSXP: return [unwrap_char(x) for x in o.data]
    if o.type in (INTSXP,LGLSXP,REALSXP): return o.data
    return o.data

def load_rdata(path):
    with gzip.open(path,'rb') as f:
        magic=f.read(5); fmt=f.read(2)
        if magic not in (b'RDX2\n',b'RDX3\n'): raise ValueError(magic)
        if fmt!=b'X\n': raise ValueError(fmt)
        r=Reader(f)
        version=r.i32(); writer=r.i32(); minr=r.i32()
        enc=None
        if version>=3:
            n=r.i32(); enc=r.readn(n).decode('ascii','replace')
        top=r.item()
        r.final_pos=r.pos() if hasattr(r,'__dict__') else None
        return top,r,(version,writer,minr,enc)

def top_env(top):
    out={}
    a=top
    while isinstance(a,RObj) and a.type==LISTSXP:
        car,cdr=a.data
        out[unwrap_char(a.tag)] = car
        a=cdr
    return out

if __name__=='__main__':
    p=sys.argv[1]
    top,r,h=load_rdata(p)
    print('header',h,'counts',r.counts,'refs',len(r.refs))
    env=top_env(top); print('objects',env.keys())
    for k,o in env.items():
        print(k,o)
        if isinstance(o,RObj):
            print(' attrs',attrs_to_dict(o.attrs).keys() if o.attrs else None)
            if o.type==VECSXP:
                ad=attrs_to_dict(o.attrs)
                names=vec_values(ad.get('names')) if 'names' in ad else None
                print(' len',len(o.data),'names',names[:10] if names else None)
                for i,col in enumerate(o.data[:3]): print(i, col.type, len(col.data) if hasattr(col.data,'__len__') else None, vec_values(col)[:5] if hasattr(vec_values(col),'__getitem__') else None)
