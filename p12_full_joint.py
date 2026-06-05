"""P12: P1投影接收P2+P5+Bridge三层反馈联合训练"""
import torch, torch.nn.functional as F, time, os, sys, re, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P2_word_char.model import WordToCharDecoder
from P5_sentence.model import SentenceSynthesis, contrastive_loss
from P6_sent_word.model import SentToWordsDecoder
from P8_char_sent.model import CharToSent

SEED=789; BATCH=200; MAX_EPOCH=500
random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
SD=os.path.join(BASE_DIR,'P1_char_word','checkpoints')

# Load P1+P11 as starting point
ck=torch.load(os.path.join(SD,'P1_best.pt'),map_location=DEVICE)
p1=CharToWordModel(ck['num_chars'],ck['num_words']).to(DEVICE)
p1.load_state_dict(ck['model_state_dict'])
p11=torch.load(os.path.join(SD,'P11_joint_best.pt'),map_location=DEVICE)
p1.output_proj.load_state_dict(p11['p1_proj'])
for name,p in p1.named_parameters():
    if 'output_proj' in name: p.requires_grad=True
    else: p.requires_grad=False
print(f'P1+P11 loaded | P2_test={p11.get("p2_test",0):.4%}')

# Load P2 from P11, keep trainable
p2=WordToCharDecoder().to(DEVICE)
p2.load_state_dict(p11['p2'])
print(f'P2 loaded')

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
all_pairs=[[c2i[c1],c2i[c2]] for _,c1,c2 in entries]
N=len(all_pairs); n_b=(N+BATCH-1)//BATCH
test_info=[(all_pairs[i],i) for i in range(N-200,N)]

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
print(f'Data: {N} words, {len(sentences)} sents')

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

# Pre-compute bridge data (char vecs + P5 sentence targets + word vecs)
print('Pre-computing bridge data...')
bd=[]
for wi,(w,r) in enumerate(sentences):
    vs=[ew(x) for x in w]
    if None in vs: continue
    cv=gcv(''.join(w))
    if cv is None: continue
    bd.append((cv,torch.stack(vs)))  # (char_vecs, word_vecs)
    if wi%500==0: print(f'  {wi}/{len(sentences)}',flush=True)
print(f'Bridge data: {len(bd)} pre-computed')

# P5 loaded frozen for sentence targets
p5=SentenceSynthesis().to(DEVICE)
p5.load_state_dict(torch.load(os.path.join(SD,'P5_best.pt'),map_location=DEVICE)['model_state_dict'])
for p in p5.parameters(): p.requires_grad=False; p5.eval()

# Pre-compute P5 sentence vectors for bridge data
print('Pre-computing P5 targets...')
for i,(cv, wv_t) in enumerate(bd):
    sv=p5(wv_t,torch.arange(min(3,len(wv_t)),device=DEVICE)%3,last_loss=0.0)
    bd[i]=(cv,sv,wv_t[:3])
    if i%500==0: print(f'  {i}/{len(bd)}',flush=True)
print(f'Bridge data ready: {len(bd)}')

# Optimizer: P1_proj + P2 + P8 + P6 all trainable
p8=CharToSent(max_len=15).to(DEVICE); p6=SentToWordsDecoder(max_words=5).to(DEVICE)
opt=torch.optim.Adam([
    {'params':p1.output_proj.parameters(),'lr':0.0002},
    {'params':p2.parameters(),'lr':0.0005},
    {'params':list(p8.parameters())+list(p6.parameters()),'lr':0.002},
],weight_decay=1e-5)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=0.5,patience=10,min_lr=1e-6)

ll=1.0; best_wc=0.0; best_ep=0; es_c=0; t0=time.time()

for ep in range(1,MAX_EPOCH+1):
    el_p1=0.0; el_p2=0.0; el_br=0.0; nb=0
    # Phase 1: P1+P2 (1000 word pairs足够投影适应)
    perm=torch.randperm(N)[:1000]
    for bi in range(0,1000,BATCH):
        s=bi; e=min(s+BATCH,1000); idxs=perm[s:e]
        pids=torch.tensor([all_pairs[i] for i in idxs],device=DEVICE)

        # P1 word disrimination
        preds=p1(pids,last_loss=ll); targets=p1.word_table[torch.tensor([i for i in idxs],device=DEVICE)]
        L1=(1.0-F.cosine_similarity(preds,targets,dim=-1).mean())**2

        # P2 char reconstruction
        _,_,full=p1.get_char_vectors(pids)
        rc1=p1.project_char(full[:,0,:]); rc2=p1.project_char(full[:,1,:])
        wv=p1(pids,last_loss=ll)
        pc1,pc2=p2(wv,last_loss=ll)
        L2=((1.0-F.cosine_similarity(pc1,rc1,dim=-1).mean())**2+(1.0-F.cosine_similarity(pc2,rc2,dim=-1).mean())**2)/2

        loss=L1+L2; opt.zero_grad(); loss.backward(); opt.step()
        el_p1+=L1.item(); el_p2+=L2.item(); nb+=1; ll=loss.item()
        torch.cuda.empty_cache()

    # Phase 2: Bridge feedback (2000句, char vecs预计算)
    random.shuffle(bd)
    for cv,sv_t,wv_t in bd[:2000]:
        sp=p8(cv,last_loss=ll); wp=p6(sp.unsqueeze(0),last_loss=ll)[0]
        nw=min(len(wp),len(wv_t))
        wcs=F.cosine_similarity(wp[:nw],wv_t[:nw],dim=-1)
        Lb=((1.0-wcs)**2).mean()
        opt.zero_grad(); Lb.backward(); opt.step()
        el_br+=Lb.item(); nb+=1; ll=Lb.item()
        torch.cuda.empty_cache()

    if ep%10==0 or ep==1:
        # P2 test
        p2.eval(); p8.eval(); p6.eval()
        tc1=[]; tc2=[]
        for ti in range(0,len(test_info),50):
            batch=[test_info[j] for j in range(ti,min(ti+50,len(test_info)))]
            tpids=torch.tensor([t[0] for t in batch],device=DEVICE)
            _,_,full=p1.get_char_vectors(tpids)
            rc1=p1.project_char(full[:,0,:]); rc2=p1.project_char(full[:,1,:])
            wv=p1(tpids,last_loss=0.0)
            pc1,pc2=p2(wv,last_loss=0.0)
            tc1.append(F.cosine_similarity(pc1,rc1,dim=-1)); tc2.append(F.cosine_similarity(pc2,rc2,dim=-1))
        p2_avg=(torch.cat(tc1).mean().item()+torch.cat(tc2).mean().item())/2

        # Bridge test
        br_wc=[]
        for w,wv_t,rl in random.sample(bd,min(50,len(bd))):
            cv=gcv(''.join(w))
            if cv is None: continue
            sp=p8(cv,last_loss=0.0); wp=p6(sp.unsqueeze(0),last_loss=0.0)[0]
            nw=min(len(wp),len(wv_t))
            br_wc.append(F.cosine_similarity(wp[:nw],wv_t[:nw],dim=-1).mean().item())
        br_avg=sum(br_wc)/len(br_wc) if br_wc else 0
        p2.train(); p8.train(); p6.train()

        score=p2_avg*0.5+br_avg*0.5  # 联合评分
        scheduler.step(p2_avg)
        lr=opt.param_groups[0]['lr']
        if score>best_wc: best_wc=score; best_ep=ep; es_c=0
        else: es_c+=10
        t='↑' if ep>10 and score>prev else ('↓' if ep>10 and score<prev else ' ')
        prev=score
        print(f'P12 E{ep:3d} | P2={p2_avg:.4%} BR={br_avg:.4%} | joint={score:.4%} {t} | best={best_wc:.4%}@{best_ep} | LR={lr:.6f} | {time.time()-t0:.0f}s')
        if ep%50==0: torch.save({'p1_proj':p1.output_proj.state_dict(),'p2':p2.state_dict(),'p8':p8.state_dict(),'p6':p6.state_dict(),'epoch':ep},os.path.join(SD,'P12_joint_best.pt'))
        if es_c>=200: print(f'CONVERGED @ {ep}'); break

print(f'\nP12 DONE: best_joint={best_wc:.4%} @ {best_ep}')
