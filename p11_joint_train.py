"""P11: P1投影+P2联合训练, 独立多目标loss + 自适应LR"""
import torch, torch.nn.functional as F, time, os, sys, re, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from utils.logger import get_log, info, metric, epoch as log_epoch, batch as log_batch
from P1_char_word.model import CharToWordModel
from P2_word_char.model import WordToCharDecoder

log = get_log("p11_joint")

parser=argparse.ArgumentParser()
parser.add_argument('--epochs',type=int,default=300,help='训练轮数')
parser.add_argument('--display',type=int,default=5,help='每N轮显示一次')
parser.add_argument('--lr_p1',type=float,default=0.0002,help='P1投影学习率')
parser.add_argument('--lr_p2',type=float,default=0.001,help='P2学习率')
args=parser.parse_args()

SEED=789; BATCH=200
random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
SD=os.path.join(BASE_DIR,'P1_char_word','checkpoints')

print(f'P11 JOINT | epochs={args.epochs} display={args.display} | lr_p1={args.lr_p1} lr_p2={args.lr_p2}')

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
N=len(all_pairs); n_batch=(N+BATCH-1)//BATCH
test_info=[(all_pairs[i],i) for i in range(N-200,N)]

# Multi-objective weights
W_P1=1.0; W_P2=3.0  # P2优化权重大于P1, 推动投影可逆性

print(f'P11 JOINT | {N} words | {n_batch} batches | W_P1={W_P1} W_P2={W_P2}')

# Load P1, freeze core, unfreeze output_proj
ck=torch.load(os.path.join(SD,'P1_best.pt'),map_location=DEVICE)
p1=CharToWordModel(ck['num_chars'],ck['num_words']).to(DEVICE)
p1.load_state_dict(ck['model_state_dict'])
for name,p in p1.named_parameters():
    if 'output_proj' in name: p.requires_grad=True
    else: p.requires_grad=False
n_proj=sum(p.numel() for p in p1.output_proj.parameters())
print(f'P1 loaded: {ck.get("top1","?")} | output_proj unfrozen: {n_proj:,} params')

# P2
p2=WordToCharDecoder().to(DEVICE)
n_p2=sum(p.numel() for p in p2.parameters())
print(f'P2: {n_p2:,} params | mod_strength={p2.mod_strength.item():.4f}')

# Joint optimizer
opt=torch.optim.Adam([
    {'params':p1.output_proj.parameters(),'lr':args.lr_p1,'name':'p1_proj'},
    {'params':p2.parameters(),'lr':args.lr_p2,'name':'p2'},
],weight_decay=1e-5)
# 自适应LR: 验证停滞5轮→减半, 最低1e-6
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=0.5,patience=5,min_lr=1e-6)
ll_p1=1.0; ll_p2=1.0; best_p2=0.0; best_ep=0; es_c=0; t0=time.time()

@torch.no_grad()
def test_p2():
    p2.eval()
    all_c1=[]; all_c2=[]
    for ti in range(0,len(test_info),50):
        batch=[test_info[j] for j in range(ti,min(ti+50,len(test_info)))]
        tpids=torch.tensor([t[0] for t in batch],device=DEVICE)
        _,_,full=p1.get_char_vectors(tpids)
        rc1=p1.project_char(full[:,0,:]); rc2=p1.project_char(full[:,1,:])
        wv=p1(tpids,last_loss=0.0)
        pc1,pc2=p2(wv,last_loss=0.0)
        all_c1.append(F.cosine_similarity(pc1,rc1,dim=-1))
        all_c2.append(F.cosine_similarity(pc2,rc2,dim=-1))
    p2.train()
    return (torch.cat(all_c1).mean().item()+torch.cat(all_c2).mean().item())/2

for ep in range(1,args.epochs+1):
    el_p1=0.0; el_p2=0.0; nb=0
    perm=torch.randperm(N)
    for bi in range(n_batch):
        s=bi*BATCH; e=min(s+BATCH,N)
        idxs=perm[s:e]
        pids=torch.tensor([all_pairs[i] for i in idxs],device=DEVICE)

        # P1 forward: word prediction
        preds=p1(pids,last_loss=ll_p1)
        targets=p1.word_table[torch.tensor([i for i in idxs],device=DEVICE)]
        p1_cos=F.cosine_similarity(preds,targets,dim=-1).mean()
        L1=(1.0-p1_cos)**2  # 独立目标: word_cos→1

        # P2 forward: char reconstruction
        _,_,full=p1.get_char_vectors(pids)
        rc1=p1.project_char(full[:,0,:]); rc2=p1.project_char(full[:,1,:])
        wv=p1(pids,last_loss=ll_p1)
        pc1,pc2=p2(wv,last_loss=ll_p2)
        c1_cos=F.cosine_similarity(pc1,rc1,dim=-1).mean()
        c2_cos=F.cosine_similarity(pc2,rc2,dim=-1).mean()
        L2=((1.0-c1_cos)**2+(1.0-c2_cos)**2)/2  # 独立目标: char_cos→1

        # Explore reg
        exp_reg=F.relu(0.02-p2.explore_state.norm())**2+F.relu(0.02-p1.explore_zone.pos_basis.norm())**2

        loss=W_P1*L1+W_P2*L2+0.01*exp_reg
        opt.zero_grad(); loss.backward(); opt.step()
        el_p1+=L1.item(); el_p2+=L2.item(); nb+=1
        ll_p1=p1_cos.item(); ll_p2=(c1_cos.item()+c2_cos.item())/2
        if bi%5==0 or bi==n_batch-1:
            a=torch.cuda.memory_allocated(DEVICE)/1024**2
            print(f'  batch {bi+1:2d}/{n_batch} | L1(word)={L1.item():.4f} L2(char)={L2.item():.4f} | P1_cos={p1_cos.item():.4f} P2_cos={((c1_cos+c2_cos)/2).item():.4f} | GPU={a:.0f}MB',flush=True)
        del pids,preds,targets,full,rc1,rc2,wv,pc1,pc2,loss
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()

    if ep%args.display==0 or ep==1:
        p2_test=test_p2()
        scheduler.step(p2_test)
        lr=opt.param_groups[0]['lr']
        if p2_test>best_p2: best_p2=p2_test; best_ep=ep; es_c=0
        else: es_c+=args.display
        t='↑' if ep>5 and p2_test>prev else ('↓' if ep>5 and p2_test<prev else ' ')
        prev=p2_test
        ms=p2.mod_strength.item(); es_n=p2.explore_state.norm().item()
        mw=p2.meta_fc[0].weight.norm().item()
        print(f'\nP11 E{ep:3d} | P2_test={p2_test:.4%} {t} | L1={el_p1/nb:.4f} L2={el_p2/nb:.4f} | best={best_p2:.4%}@{best_ep} | LR={lr:.6f} | mod_s={ms:.4f} exp_n={es_n:.1f} meta_w={mw:.1f} | {time.time()-t0:.0f}s\n')
        log_epoch(ep, P2_test=f"{p2_test:.4%}", L1=f"{el_p1/nb:.4f}", L2=f"{el_p2/nb:.4f}", best=f"{best_p2:.4%}", LR=f"{lr:.6f}")
        if ep==best_ep:  # 刚刷新best时保存
            torch.save({'p1_proj':p1.output_proj.state_dict(),'p2':p2.state_dict(),
                       'p2_test':p2_test,'epoch':ep},os.path.join(SD,'P11_joint_best.pt'))
        if es_c>=100: print(f'CONVERGED @ {ep}'); break

print(f'\nP11 DONE: best_p2={best_p2:.4%} @ epoch {best_ep}')
info(f"P11_COMPLETE best_p2={best_p2:.4%} epoch={best_ep}")
