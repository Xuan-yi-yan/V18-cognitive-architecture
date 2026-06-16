"""全链路快速验证: P1(CPU逐词)→P3-L(冻结)→P7(P5式)+P6→词"""
import torch, torch.nn.functional as F, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P3_word_attr.stack import P3AttributeStack
from P3_word_attr.p3l_linkage import P3L_AttributeLinkage, AttributeValueRegistry
from P7_cross_sent.model import P7WordRouter2048
from P6_sent_word.model import SentToWordsDecoder

# P1: 分批CPU编码, 每50词清缓存
p1_ckpt = torch.load(os.path.join(SAVE_DIR, 'P1_best.pt'), map_location=DEVICE)
p1 = CharToWordModel(p1_ckpt['num_chars'], p1_ckpt['num_words']).to(DEVICE)
p1.load_state_dict(p1_ckpt['model_state_dict']); p1.eval()
c2i = p1_ckpt['char2idx']

# 数据
pairs=[]
with open('C:/ai/P7_cross_sent/data_p7_v5.txt','r',encoding='utf-8') as f:
    for line in f:
        line=line.strip()
        if not line or line.startswith('#'): continue
        a,b=line.split('|')
        pairs.append((a.split(),b.split()))

# P3注册 (CPU)
p3_stack = P3AttributeStack()
registry = AttributeValueRegistry()
all_w=set()
for A,B in pairs: all_w.update(A); all_w.update(B)
for w in sorted(all_w):
    pkt = p3_stack.process_word(w)
    registry.register_from_packet(pkt)
for A,B in pairs:
    if len(A)>=2 and len(B)>=2:
        for pkt in p3_stack.process_sentence(A): registry.register_from_packet(pkt, level='all')
        for pkt in p3_stack.process_sentence(B): registry.register_from_packet(pkt, level='all')

# 编码句对 — 分批CPU, 立即转GPU
print("编码...")
word_cache={}
all_uniq = sorted(all_w)
batch_w=[]; batch_ids=[]
for w in all_uniq:
    c1=w[0]; c2=w[0] if len(w)==1 else w[1]
    if c1 not in c2i or c2 not in c2i: continue
    batch_w.append(w); batch_ids.append([c2i[c1],c2i[c2]])
    if len(batch_ids)>=50:
        t=torch.tensor(batch_ids,device=DEVICE)
        with torch.no_grad(): vecs = p1(t, last_loss=0.0)
        if vecs.dim()==1: vecs=vecs.unsqueeze(0)
        for bw,bv in zip(batch_w, vecs): word_cache[bw]=bv
        batch_w=[]; batch_ids=[]
        torch.cuda.empty_cache()
if batch_ids:
    with torch.no_grad(): vecs=p1(torch.tensor(batch_ids,device=DEVICE),last_loss=0.0)
    for bw,bv in zip(batch_w,vecs): word_cache[bw]=bv
del p1
print(f'编码: {len(word_cache)}词, GPU: {torch.cuda.memory_allocated()/1024**2:.0f}MB')

encoded=[]
for A,B in pairs:
    if not all(w in word_cache for w in A+B): continue
    Av_list=[word_cache[w] for w in A]
    Bv_list=[word_cache[w] for w in B]
    # 确保每个词向量都是[128]
    if any(v.shape!=torch.Size([128]) for v in Av_list+Bv_list):
        continue
    Av=torch.stack([v.clone() for v in Av_list])
    Bv=torch.stack([v.clone() for v in Bv_list])
    A_pkts = p3_stack.process_sentence(A)
    B_pkts = p3_stack.process_sentence(B)
    encoded.append((Av,Bv,A,B,A_pkts,B_pkts))
print(f'有效句对: {len(encoded)}, GPU: {torch.cuda.memory_allocated()/1024**2:.0f}MB')

# 模型 (P3-L暂时不需要训练, 只验证P7+P6链路)
p7 = P7WordRouter2048().to(DEVICE)
p6 = SentToWordsDecoder(max_words=16).to(DEVICE)

opt_p7 = torch.optim.Adam(p7.parameters(), lr=0.005)
opt_p6 = torch.optim.Adam(p6.parameters(), lr=0.005)

long_idx=max(range(len(encoded)),key=lambda i: len(encoded[i][2]))
A_v,B_v,A_w,B_w,A_p,B_p=encoded[long_idx]

# 第1步: P7快训2轮
print("P7快训...")
for ep in range(1,11):  # 10轮
    for Av,Bv,_,_,_,_ in encoded:
        _, sv, _, _, _ = p7(Av, Bv, last_loss=0.0)
        B_sent = p7.sent_proj(Bv.sum(dim=0, keepdim=True))
        loss = (1.0-(F.normalize(sv.unsqueeze(0),dim=-1)*F.normalize(B_sent,dim=-1)).sum()).abs()
        opt_p7.zero_grad(); loss.backward(); opt_p7.step()
    if ep<=2 or ep%5==0: print(f'  P7 E{ep}/10 OK')
p7.eval()

# 第2步: P6训3轮 (P7冻结)
print("P6训练...")
for ep in range(1, 21):  # 20轮
    total=0.0; n=0
    for Av,Bv,Aw,Bw,_,_ in encoded:
        with torch.no_grad():
            _, sv, _, _, _ = p7(Av, Bv, last_loss=0.0)
        pred_w = p6(sv.unsqueeze(0))[:,:len(Bw),:]
        cos = F.cosine_similarity(pred_w, Bv.unsqueeze(0), dim=-1).mean()
        loss = 1.0 - cos
        total+=loss.item(); n+=1
        opt_p6.zero_grad(); loss.backward(); opt_p6.step()

    with torch.no_grad():
        _, sv, _, _, _ = p7(A_v, B_v, last_loss=0.0)
        pred_w = p6(sv.unsqueeze(0))[:,:len(B_w),:]
        pred_n=F.normalize(pred_w.squeeze(0),dim=-1)
        true_n=F.normalize(B_v,dim=-1)
        c=torch.mm(pred_n,true_n.T).argmax(dim=-1)
        pred=[B_w[c[i].item()] for i in range(len(B_w))]
        ok=sum(1 for p,t in zip(pred,B_w) if p==t)
    if ep<=3 or ep%5==0:
        print(f'E{ep}/20 | loss={total/n:.4f} | pred={pred} ({ok}/{len(B_w)})')
print(f'GPU: {torch.cuda.memory_allocated()/1024**2:.0f}MB')
