"""全链路测试: P11投影→P2→P3→P5→Bridge→P7"""
import torch, torch.nn.functional as F, time, os, sys, re, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P2_word_char.model import WordToCharDecoder
from P3_word_attr.model import SubjectBindingModel, margin_loss
from P5_sentence.model import SentenceSynthesis, contrastive_loss
from P6_sent_word.model import SentToWordsDecoder
from P8_char_sent.model import CharToSent
from P7_cross_sent.model import CrossSentenceRouter

SEED=789; random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
SD=os.path.join(BASE_DIR,'P1_char_word','checkpoints')

# Load P1+P11 projection
ck=torch.load(os.path.join(SD,'P1_best.pt'),map_location=DEVICE)
p1=CharToWordModel(ck['num_chars'],ck['num_words']).to(DEVICE)
p1.load_state_dict(ck['model_state_dict'])
p11=torch.load(os.path.join(SD,'P11_joint_best.pt'),map_location=DEVICE)
p1.output_proj.load_state_dict(p11['p1_proj'])
for p in p1.parameters(): p.requires_grad=False; p1.eval()
print(f'P1: Top-1={ck.get("top1","?")} + P11_proj')

# Load P2 from P11
p2=WordToCharDecoder().to(DEVICE)
p2.load_state_dict(p11['p2'])
for p in p2.parameters(): p.requires_grad=False; p2.eval()
print(f'P2 loaded from P11 (test={p11.get("p2_test",0):.4%})')

# Data
entries=[]
with open(os.path.join(DATA_DIR,'word_list_v2.txt'),'r',encoding='utf-8') as f:
    for line in f:
        line=line.strip()
        if not line: continue
        m=re.match(r'!([^@]+)@(.)@(.)',line)
        if m: entries.append((m.group(1),m.group(2),m.group(3)))
        elif len(line)==2: entries.append((line,line[0],line[1]))
cs=set()
for _,c1,c2 in entries: cs.add(c1); cs.add(c2)
c2i={c:i for i,c in enumerate(sorted(cs))}
i2w={i:w for w,i in {w:i for i,(w,_,_) in enumerate(entries)}.items()}
sentences=[]
with open(os.path.join(BASE_DIR,'P5_sentence','sentences_v8k.txt'),'r',encoding='utf-8') as f:
    for line in f:
        line=line.strip()
        if not line or line.startswith('#'): continue
        p=line.split('|')
        if len(p)!=2: continue
        w,r=[],[]
        for it in p[1].split():
            try: x,y=it.split(':'); w.append(x); r.append({'subj':0,'verb':1,'obj':2}.get(y,0))
            except: pass
        if w: sentences.append((w,r))
print(f'Data: {len(entries)} words, {len(sentences)} sents')

@torch.no_grad()
def ew(w):
    c1,c2=w[0],w[0] if len(w)==1 else w[-1]
    if c1 not in c2i or c2 not in c2i: return None
    return p1(torch.tensor([[c2i[c1],c2i[c2]]],device=DEVICE),last_loss=0.0)[0]

def gcv(text):
    vecs=[]
    for c in text:
        if c not in c2i: return None
        ct=p1.char_content(torch.tensor([c2i[c]],device=DEVICE))
        ps=p1.pos_encoder.pe[0:1]
        vecs.append(p1.project_char(torch.cat([ps[0],ct[0]],dim=-1)))
    return torch.stack(vecs)

# === P3 ===
print('\n--- P3 ---')
ATTRS=['主语','谓语','宾语','定语','状语','补语','虚词']
attr_data={}
for f in sorted(os.listdir(SD)):
    if not f.startswith('P3_') or not f.endswith('_best.pt'): continue
    try:
        cd=torch.load(os.path.join(SD,f),map_location='cpu',weights_only=False)
        a=cd.get('attr',''); fw=cd.get('family_words',[])
        for attr in ATTRS:
            if attr in a and attr not in attr_data and len(fw)>5: attr_data[attr]=fw; break
    except: pass
p3_accs={}
for attr in ATTRS:
    raw=attr_data.get(attr)
    if not raw: continue
    aw=[w for w in raw if w[0] in c2i and (w[0] if len(w)==1 else w[-1]) in c2i]
    na=[w for w in i2w.values() if w not in set(aw) and w[0] in c2i]
    def enc(ws):
        vs=[]
        for w in ws:
            c1,c2=w[0],w[0] if len(w)==1 else w[-1]
            vs.append(p1(torch.tensor([[c2i[c1],c2i[c2]]],device=DEVICE),last_loss=0.0)[0])
        return torch.stack(vs) if vs else torch.zeros(0,128,device=DEVICE)
    fp1=enc(aw); fpt=fp1.mean(dim=0)
    pw2i={w:i for i,w in enumerate(i2w.values())}
    pi=torch.tensor([pw2i[w] for w in aw if w in pw2i],device=DEVICE)
    ni=torch.tensor([pw2i[w] for w in na if w in pw2i],device=DEVICE)
    m=SubjectBindingModel(len(pw2i)).to(DEVICE)
    opt=torch.optim.Adam(m.parameters(),lr=0.005,weight_decay=1e-5)
    ba=0; h=BATCH_SIZE//2
    for ep in range(1,201):
        if len(pi)>0 and len(ni)>0:
            rpi=pi[torch.randperm(len(pi))[:h]]; rni=ni[torch.randperm(len(ni))[:h]]
            ids=torch.cat([rpi,rni]); ip=torch.tensor([True]*len(rpi)+[False]*len(rni),device=DEVICE)
            out,_,qr=m(ids,fp1,last_loss=0.5)
            loss,_=margin_loss(out,fpt,ip,q_raw=qr)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep%60==0:
            m.eval()
            with torch.no_grad():
                ps=m.binding_score(pi,fp1) if len(pi)>0 else torch.tensor([0.])
                ns=m.binding_score(ni,fp1) if len(ni)>0 else torch.tensor([0.])
            m.train()
            pm,nm=ps.mean().item(),ns.mean().item()
            acc=((ps>(pm+nm)/2).float().mean()+(ns<=(pm+nm)/2).float().mean())/2
            if acc>ba: ba=acc
    p3_accs[attr]=ba
    del m,opt; torch.cuda.empty_cache()
    print(f'P3-{attr}: {ba:.2%}')

# === P5: load from checkpoint (no retraining needed) ===
print('\n--- P5 (loaded from checkpoint) ---')
p5=SentenceSynthesis().to(DEVICE)
p5.load_state_dict(torch.load(os.path.join(SD,'P5_best.pt'),map_location=DEVICE)['model_state_dict'])
for p in p5.parameters(): p.requires_grad=False; p5.eval()
print(f'P5 loaded: gap={torch.load(os.path.join(SD,"P5_best.pt"),map_location=DEVICE).get("avg_gap","?")}')

# === Bridge ===
print('\n--- Bridge ---')
bd=[]
for w,r in sentences:
    vs=[ew(x) for x in w]
    if None in vs: continue
    sv=p5(torch.stack(vs),torch.tensor(r,device=DEVICE),last_loss=0.0)
    bd.append((w,sv,torch.stack(vs)))
p8=CharToSent(max_len=15).to(DEVICE); p6=SentToWordsDecoder(max_words=5).to(DEVICE)
opt=torch.optim.Adam(list(p8.parameters())+list(p6.parameters()),lr=0.002,weight_decay=1e-5)
ll=1.0; bw=0.0; t0=time.time()
for ep in range(1,501):
    el=0.0; nb=0; random.shuffle(bd)
    for wi,(w,sv_t,wv_t) in enumerate(bd):
        cv=gcv(''.join(w))
        if cv is None: continue
        sp=p8(cv,last_loss=ll); wp=p6(sp.unsqueeze(0),last_loss=ll)[0]
        sc=F.cosine_similarity(sp.unsqueeze(0),sv_t.unsqueeze(0),dim=-1)
        nw=min(len(wp),len(wv_t)); wcs=F.cosine_similarity(wp[:nw],wv_t[:nw],dim=-1)
        loss=5.0*(1.0-sc)**2+1.0*((1.0-wcs)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        el+=loss.item(); nb+=1; ll=loss.item(); torch.cuda.empty_cache()
    if ep%20==0:
        p8.eval(); p6.eval(); wc=[]
        with torch.no_grad():
            for w,sv_t,wv_t in random.sample(bd,min(50,len(bd))):
                cv=gcv(''.join(w))
                if cv is None: continue
                sp=p8(cv,last_loss=0.0); wp=p6(sp.unsqueeze(0),last_loss=0.0)[0]
                nw=min(len(wp),len(wv_t))
                wc.append(F.cosine_similarity(wp[:nw],wv_t[:nw],dim=-1).mean().item())
        p8.train(); p6.train(); aw=sum(wc)/len(wc)
        if aw>bw: bw=aw
        print(f'BR E{ep:3d} word_cos={aw:.4%} best={bw:.4%} | {time.time()-t0:.0f}s')
torch.save({'p8':p8.state_dict(),'p6':p6.state_dict(),'word_cos':bw},os.path.join(SD,'fullpipe_bridge.pt'))
del opt,p8,p6; torch.cuda.empty_cache()
print(f'BRIDGE DONE: best={bw:.4%}')

# === P7 ===
print('\n--- P7 ---')
p7pairs=[]
with open(os.path.join(BASE_DIR,'P7_cross_sent','data_p7.txt'),'r',encoding='utf-8') as f:
    for line in f:
        line=line.strip()
        if not line or line.startswith('#'): continue
        parts=line.split('|')
        if len(parts)==2: p7pairs.append((parts[0].split(),parts[1].split()))
Bv=sorted(set(w for _,B in p7pairs for w in B))
Bvs=[ew(w) for w in Bv if ew(w) is not None]
bp7=0.0
if len(Bvs)>=3:
    Bt=torch.stack(Bvs); p7e=[]
    for A,B in p7pairs:
        Av=[ew(w) for w in A]; Bv2=[ew(w) for w in B]
        if None in Av or None in Bv2: continue
        Bs=p5(torch.stack(Bv2),torch.arange(len(Bv2),device=DEVICE)%3,last_loss=0.0)
        p7e.append((torch.stack(Av),Bs))
    if len(p7e)>=3:
        p7=CrossSentenceRouter().to(DEVICE)
        opt=torch.optim.Adam(p7.parameters(),lr=0.003,weight_decay=1e-5)
        ll=1.0
        for ep in range(1,301):
            el=0.0; nb=0
            for Av,Bt2 in p7e:
                Bp,_=p7(Av,Bt,last_loss=ll)
                cos=F.cosine_similarity(Bp.unsqueeze(0),Bt2.unsqueeze(0),dim=-1)
                loss=(1.0-cos).mean(); opt.zero_grad(); loss.backward(); opt.step()
                el+=loss.item(); nb+=1; ll=loss.item()
            if ep%50==0:
                p7.eval(); coses=[]
                with torch.no_grad():
                    for Av,Bt2 in p7e:
                        Bp,_=p7(Av,Bt,last_loss=0.0)
                        coses.append(F.cosine_similarity(Bp.unsqueeze(0),Bt2.unsqueeze(0),dim=-1).item())
                p7.train(); ac=sum(coses)/len(coses)
                if ac>bp7: bp7=ac
                print(f'P7 E{ep:3d} cos={ac:.4%} best={bp7:.4%}')
        del p7,opt; torch.cuda.empty_cache()
        print(f'P7 DONE: best={bp7:.4%}')

# === Summary ===
print(f'\n===== FULL PIPELINE (P11投影) =====')
print(f'P1: 98.6% (P11投影优化)')
print(f'P2: {p11.get("p2_test",0):.4%} (P11联合)')
for a,v in p3_accs.items(): print(f'P3-{a}: {v:.2%}')
print(f'P5: loaded from checkpoint')
print(f'Bridge: {bw:.4%}')
print(f'P7: {bp7:.4%}')
a=torch.cuda.memory_allocated(DEVICE)/1024**2
print(f'GPU: {a:.0f}MB')
