"""P2 深训: 500/批次, 逐批loss, 每轮清缓存"""
import torch, torch.nn.functional as F, time, os, sys, re, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from utils.logger import get_log, info, epoch as log_epoch

log = get_log("p2_deep")
from P1_char_word.model import CharToWordModel
from P2_word_char.model import WordToCharDecoder

SEED=789; BATCH=200; MAX_EPOCH=500
random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
SD=os.path.join(BASE_DIR,'P1_char_word','checkpoints')

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
N=len(all_pairs)
train_pairs=all_pairs[:N-200]
test_info=[(all_pairs[i],i) for i in range(N-200,N)]
n_batches=(len(train_pairs)+BATCH-1)//BATCH

print(f'P2 DEEP | {N} words | {len(train_pairs)} train / 200 test | {n_batches} batches x {BATCH} | max {MAX_EPOCH} epochs')
ck=torch.load(os.path.join(SD,'P1_best.pt'),map_location=DEVICE)
p1=CharToWordModel(ck['num_chars'],ck['num_words']).to(DEVICE)
p1.load_state_dict(ck['model_state_dict'])
for p in p1.parameters(): p.requires_grad=False; p1.eval()
print(f'P1: Top-1={ck.get("top1","?")}')

p2=WordToCharDecoder().to(DEVICE)
opt=torch.optim.Adam(p2.parameters(),lr=0.001,weight_decay=1e-5)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=0.5,patience=8,min_lr=1e-6)
ll=1.0; best_test=0.0; best_ep=0; es_c=0; t0=time.time()

@torch.no_grad()
def test_p2():
    p2.eval()
    all_c1=[]; all_c2=[]
    for ti in range(0, len(test_info), 50):  # batch test into 50
        batch=[test_info[j] for j in range(ti, min(ti+50, len(test_info)))]
        tpids=torch.tensor([t[0] for t in batch],device=DEVICE)
        _,_,full=p1.get_char_vectors(tpids)
        rc1=p1.project_char(full[:,0,:]); rc2=p1.project_char(full[:,1,:])
        wv=p1(tpids,last_loss=0.0)
        pc1,pc2=p2(wv,last_loss=0.0)
        all_c1.append(F.cosine_similarity(pc1,rc1,dim=-1))
        all_c2.append(F.cosine_similarity(pc2,rc2,dim=-1))
        del tpids,full,rc1,rc2,wv,pc1,pc2
    p2.train()
    c1=torch.cat(all_c1).mean().item()
    c2=torch.cat(all_c2).mean().item()
    return (c1+c2)/2, c1, c2

for ep in range(1,MAX_EPOCH+1):
    el1=0.0; el2=0.0; nb=0
    perm=torch.randperm(len(train_pairs))
    for bi in range(n_batches):
        s=bi*BATCH; e=min(s+BATCH,len(train_pairs))
        idxs=perm[s:e]
        pids=torch.tensor([train_pairs[i] for i in idxs],device=DEVICE)
        with torch.no_grad():
            _,_,full=p1.get_char_vectors(pids)
            rc1=p1.project_char(full[:,0,:]); rc2=p1.project_char(full[:,1,:])
            wv=p1(pids,last_loss=ll)
        pc1,pc2=p2(wv,last_loss=ll)
        s1=F.cosine_similarity(pc1,rc1,dim=-1).mean()
        s2=F.cosine_similarity(pc2,rc2,dim=-1).mean()
        L1=(1.0-s1).item(); L2=(1.0-s2).item()
        loss=(1.0-(s1+s2)/2.0)+0.001*F.relu(0.1-p2.explore_state.norm())
        opt.zero_grad(); loss.backward(); opt.step()
        el1+=L1; el2+=L2; nb+=1; ll=loss.item()
        del pids,full,rc1,rc2,wv,pc1,pc2,loss
        torch.cuda.empty_cache()  # 逐批硬清
        if bi%3==0 or bi==n_batches-1:
            a=torch.cuda.memory_allocated(DEVICE)/1024**2
            print(f'  batch {bi+1:2d}/{n_batches} | L1={L1:.4f} L2={L2:.4f} | GPU={a:.0f}MB',flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if ep%5==0 or ep==1:
        test_avg,tc1,tc2=test_p2()
        scheduler.step(test_avg)
        lr=opt.param_groups[0]['lr']
        if test_avg>best_test: best_test=test_avg; best_ep=ep; es_c=0
        else: es_c+=5
        t='↑' if ep>5 and test_avg>prev else ('↓' if ep>5 and test_avg<prev else ' ')
        prev=test_avg
        exp_n=p2.explore_state.norm().item(); mw=p2.meta_fc[0].weight.norm().item()
        print(f'\nP2 E{ep:3d} | test={test_avg:.4%} {t} (c1={tc1:.4%} c2={tc2:.4%}) | L1avg={el1/nb:.4f} L2avg={el2/nb:.4f} | best={best_test:.4%}@{best_ep} | LR={lr:.6f} | exp_n={exp_n:.4f} meta_w={mw:.2f} | {time.time()-t0:.0f}s\n')
        if test_avg>best_test:
            torch.save({'model_state_dict':p2.state_dict(),'test_cos':test_avg,'epoch':ep},os.path.join(SD,'P2_deep_best.pt'))
        if es_c>=100: print(f'CONVERGED @ {ep}'); break

print(f'\nP2 DEEP DONE: test={best_test:.4%} @ epoch {best_ep}')
a=torch.cuda.memory_allocated(DEVICE)/1024**2
print(f'GPU final: {a:.0f}MB')
