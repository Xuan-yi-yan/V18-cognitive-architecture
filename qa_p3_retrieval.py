"""QA: P3属性+检索头→P5合成答案句"""
import torch, torch.nn.functional as F, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P3_word_attr.model import SubjectBindingModel
from P3_word_attr.retrieval import P3RetrievalHead
from P5_sentence.model import SentenceSynthesis

SEED=789; torch.manual_seed(SEED)
SD=os.path.join(BASE_DIR,'P1_char_word','checkpoints')

# Load P1
ck=torch.load(os.path.join(SD,'P1_best.pt'),map_location=DEVICE)
p1=CharToWordModel(ck['num_chars'],ck['num_words']).to(DEVICE)
p1.load_state_dict(ck['model_state_dict'])
p11p=os.path.join(SD,'P11_joint_best.pt')
if os.path.exists(p11p):
    p1.output_proj.load_state_dict(torch.load(p11p,map_location=DEVICE)['p1_proj'])
for p in p1.parameters(): p.requires_grad=False; p1.eval()
c2i=ck['char2idx']; i2w=ck['idx2word']; N=ck['num_words']
word_table=p1.get_all_reference_vectors(DEVICE)  # [N,128]

# Load P3 attribute models
ATTRS=['主语','谓语','宾语','定语','状语','补语','虚词']
p3_models={}
attr_words={}
for attr in ATTRS:
    for f in os.listdir(SD):
        if f'P3_{attr}_best' in f and f.endswith('.pt'):
            ck3=torch.load(os.path.join(SD,f),map_location=DEVICE,weights_only=False)
            if 'p3_word2id' in ck3:
                m=SubjectBindingModel(len(ck3['p3_word2id'])).to(DEVICE)
                m.load_state_dict(ck3['model'])
                for p in m.parameters(): p.requires_grad=False; m.eval()
                p3_models[attr]=m
                attr_words[attr]=ck3.get('family_words',[])
                break
attr_words={k:set(v) for k,v in attr_words.items()}  # list→set for fast lookup
print(f'P3 loaded: {list(p3_models.keys())} | attr sets: {[(k,len(v)) for k,v in attr_words.items()]}')

# Load P5
p5=SentenceSynthesis().to(DEVICE)
p5.load_state_dict(torch.load(os.path.join(SD,'P5_best.pt'),map_location=DEVICE)['model_state_dict'])
for p in p5.parameters(): p.requires_grad=False; p5.eval()

@torch.no_grad()
def ew(w):
    if len(w)==1: c1=c2=w[0]
    else: c1,c2=w[0],w[-1]
    if c1 not in c2i or c2 not in c2i: return None
    return p1(torch.tensor([[c2i[c1],c2i[c2]]],device=DEVICE),last_loss=0.0)[0]

@torch.no_grad()
def p3_get_attr(word, wv):
    """P3: 返回word最可能的属性"""
    best_attr=None; best_score=-1
    for attr,model in p3_models.items():
        if attr not in attr_words or len(attr_words[attr])<3: continue
        words=list(attr_words[attr])[:50]
        family_vecs=[ew(w) for w in words if ew(w) is not None]
        if len(family_vecs)<3: continue
        family=torch.stack(family_vecs)
        score=model.binding_score(
            torch.tensor([0],device=DEVICE),family
        ).item()
        if score>best_score: best_score=score; best_attr=attr
    return best_attr,best_score

@torch.no_grad()
def retrieve_replacement(wv, attr, word_table_n, i2w, attr_words, topk=10):
    """检索: 找word_table中与wv最相似且属于attr族群的词"""
    sims=torch.mm(F.normalize(wv.unsqueeze(0),dim=-1),word_table_n.T).squeeze(0)
    top=torch.topk(sims,min(topk*20,len(sims)))  # 先取更多候选
    results=[]
    for idx,sim in zip(top.indices.tolist(),top.values.tolist()):
        w=i2w[idx]
        # 优先选属于目标属性族的词
        if w in attr_words.get(attr,set()):
            results.append((w,sim))
            if len(results)>=topk: break
    # Fallback: 没找到足够的, 补充相似词
    if len(results)<3:
        for idx,sim in zip(top.indices.tolist(),top.values.tolist()):
            w=i2w[idx]
            if (w,sim) not in results and len(results)<topk:
                results.append((w,sim))
    return results

word_table_n=F.normalize(word_table,dim=-1)

# Question words that trigger replacement
question_words={'什么':'宾语','谁':'主语','哪里':'宾语','怎么':'状语','什么时候':'状语'}

tests=[
    (['你','喜欢','什么'], '你 喜欢 ___'),
    (['谁','教','学生'], '___ 教 学生'),
    (['老师','教','什么'], '老师 教 ___'),
    (['他','去','哪里'], '他 去 ___'),
]

for q_words,slot_desc in tests:
    print(f'\nQ: {q_words}')
    print(f'  槽位: {slot_desc}')

    # Encode + P3 tag + retrieve
    replacements=[]
    for w in q_words:
        wv=ew(w)
        if wv is None: print(f'  [SKIP] {w} not encodable'); continue

        if w in question_words:
            # 疑问词 → 检索替代
            target_attr=question_words[w]
            candidates=retrieve_replacement(wv, target_attr, word_table_n, i2w, attr_words, topk=10)
            # Filter by rough attribute match
            print(f'  [{w}] 疑问词(需{target_attr}) → Top candidates: {[(c,round(s,3)) for c,s in candidates[:5]]}')
            # Pick best candidate NOT in question
            for cand,sim in candidates:
                if cand not in q_words:
                    replacements.append(cand)
                    print(f'  -> PICK: {cand}')
                    break
        else:
            # 保留原词
            replacements.append(w)
            print(f'  [{w}] 保留')

    if len(replacements)>=2:
        print(f'\n  >>> ANSWER: {replacements}')
