"""QA推理: 问题→P8→P7→P6→P1→答案词"""
import torch, torch.nn.functional as F, os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P7_cross_sent.model import CrossSentenceRouter

SD=os.path.join(BASE_DIR,'P1_char_word','checkpoints')

# Load P1
ck=torch.load(os.path.join(SD,'P1_best.pt'),map_location=DEVICE)
p1=CharToWordModel(ck['num_chars'],ck['num_words']).to(DEVICE)
p1.load_state_dict(ck['model_state_dict'])
p11p=os.path.join(SD,'P11_joint_best.pt')
if os.path.exists(p11p):
    p11=torch.load(p11p,map_location=DEVICE)
    p1.output_proj.load_state_dict(p11['p1_proj'])
c2i=ck['char2idx']; i2w=ck['idx2word']
for p in p1.parameters(): p.requires_grad=False; p1.eval()

# Load P7
p7_ck=torch.load(os.path.join(SD,'P7_v2_best.pt'),map_location=DEVICE)
p7=CrossSentenceRouter().to(DEVICE)
p7.load_state_dict(p7_ck['model_state_dict'])
B_vocab=p7_ck['B_vocab']
for p in p7.parameters(): p.requires_grad=False; p7.eval()

# Build B word table (P1 encoded)
@torch.no_grad()
def ew(w):
    if len(w)==1: c1=c2=w[0]
    else: c1,c2=w[0],w[-1]
    if c1 not in c2i or c2 not in c2i: return None
    return p1(torch.tensor([[c2i[c1],c2i[c2]]],device=DEVICE),last_loss=0.0)[0]
B_vecs=[ew(w) for w in B_vocab if ew(w) is not None]
B_table=torch.stack(B_vecs)
B_n=F.normalize(B_table,dim=-1)

ref=p1.get_all_reference_vectors(DEVICE)
ref_n=F.normalize(ref,dim=-1)

print(f'Loaded: P1={len(c2i)}chars, P7 B_vocab={len(B_vocab)}words\n')

def qa(question_words, topk=3):
    print(f'Q: {question_words}')

    # Encode question words via P1 (question_words is a list)
    q_words=[]
    for w in question_words:
        wv=ew(w)
        if wv is None:
            if len(w)==2 and w[0] in c2i and w[1] in c2i:
                pid=torch.tensor([[c2i[w[0]],c2i[w[1]]]],device=DEVICE)
                wv=p1(pid,last_loss=0.0)[0]
            else:
                print(f'  [SKIP] word "{w}" not encodable')
                continue
        q_words.append((w,wv))

    if len(q_words)==0: print('  [FAIL] no words encodable'); return
    print(f'  Encoded: {[w for w,_ in q_words]}')

    # P7: question words → answer sentence vector
    q_wvs=torch.stack([wv for _,wv in q_words])
    B_pred,attn=p7(q_wvs,B_table,last_loss=0.0)
    B_pred_n=F.normalize(B_pred,dim=-1)

    # Match answer sentence to B_vocab word vectors
    # Each B_vocab word → cosine with B_pred
    b_sims=torch.mm(B_pred_n.unsqueeze(0),B_n.T).squeeze(0)
    top_b=torch.topk(b_sims,min(topk,len(B_vocab)))
    print(f'  P7 top-B words: {[(B_vocab[i],round(b_sims[i].item(),3)) for i in top_b.indices.tolist()]}')

    # Also try: match B_pred to full P1 word table
    p1_sims=torch.mm(B_pred_n.unsqueeze(0),ref_n.T).squeeze(0)
    top_p1=torch.topk(p1_sims,min(topk,len(ref)))
    print(f'  P1 top matches: {[(i2w[i],round(p1_sims[i].item(),3)) for i in top_p1.indices.tolist()]}')

    # Attention: which question words attend to which B words
    attn_avg=attn.mean(dim=0)  # [nA, nB]
    for ai,(w,_) in enumerate(q_words):
        top_attn=torch.topk(attn_avg[ai],min(3,len(B_vocab)))
        print(f'  {w} attends to: {[(B_vocab[i],round(attn_avg[ai][i].item(),3)) for i in top_attn.indices.tolist()]}')

# Test questions
tests=[
    ["你","喜欢","什么"],
    ["谁","教","学生"],
    ["老师","教","什么"],
    ["他","去","哪里"],
    ["你","怎么","学习"],
    ["学生","学习","知识"],
]
for q in tests:
    try: qa(q)
    except Exception as e: print(f'  [ERROR] {e}')
    print()
