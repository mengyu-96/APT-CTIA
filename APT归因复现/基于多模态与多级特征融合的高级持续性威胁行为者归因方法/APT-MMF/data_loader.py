import glob,hashlib,json,os,re,urllib.request,warnings,ipaddress,tempfile
from collections import Counter,defaultdict
import sys,types
if 'dgl.graphbolt' not in sys.modules:
 sys.modules['dgl.graphbolt']=types.ModuleType('dgl.graphbolt')
import dgl,numpy as np,torch
from transformers import AutoModel,AutoTokenizer
from model_utils import NODE_TYPE
try: import pdfplumber
except: pdfplumber=None
try: from pypdf import PdfReader
except: PdfReader=None
warnings.filterwarnings('ignore')
ATTACK_URL='https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json'
HF_MODEL=os.environ.get('HF_MODEL','bert-base-cased');HF_ENDPOINT=os.environ.get('HF_ENDPOINT','https://hf-mirror.com')
IOC_TYPES=['Registry','FilePath','FileName','Email','URL','Domain','IP','Tactic','Technique','Vulnerability','Malware']
W={'Registry':1.0,'FilePath':1.0,'FileName':0.8,'Email':1.2,'URL':1.2,'Domain':1.2,'IP':1.1,'Tactic':1.4,'Technique':1.5,'Vulnerability':1.3,'Malware':1.3}
NOISY_DOMAIN_SUFFIX=('microsoft.com','github.com','virustotal.com','wikipedia.org','mitre.org','google.com')
FEATURE_CACHE_VERSION='data_loader_v10_masked_mean_pool_title_entities'
ENABLE_SEGMENT_AUG=os.environ.get('APT_ENABLE_SEGMENT_AUG','0').lower() in ('1','true','yes','on')
SEGMENT_MIN_CHARS=int(os.environ.get('APT_SEGMENT_MIN_CHARS','7000'))
SEGMENT_CHARS=int(os.environ.get('APT_SEGMENT_CHARS','5000'))
SEGMENT_STRIDE=int(os.environ.get('APT_SEGMENT_STRIDE','3500'))
MAX_SEGMENTS_PER_REPORT=int(os.environ.get('APT_MAX_SEGMENTS_PER_REPORT','3'))
SPLIT_SEED=int(os.environ.get('APT_SPLIT_SEED','72'))
INCLUDE_TACTIC_MP=os.environ.get('APT_INCLUDE_TACTIC_MP','0').lower() in ('1','true','yes','on')
def _pad(x,d):
 x=torch.as_tensor(x,dtype=torch.float32);return x[:d] if x.ndim==1 and x.numel()>=d else (torch.cat([x,torch.zeros(d-x.numel())]) if x.ndim==1 else (x[:,:d] if x.shape[1]>=d else torch.cat([x,torch.zeros((x.shape[0],d-x.shape[1]))],1)))
def _ok(p): return os.path.isfile(p) and os.path.getsize(p)>0
def _dataset(base):
 for p in [os.path.join(base,'APT-MMF','dataset'),os.path.join(base,'dataset'),os.path.join(base,'Dataset'),os.path.join(base,'data'),os.path.join(base,'Data')]:
  if os.path.isdir(p): return p
 raise RuntimeError('Dataset directory not found. Expected APT-MMF/dataset or project-root dataset/data.')
def _attack(base):
 for p in [os.path.join(base,'external_knowledge','attack','enterprise-attack.json'),os.path.join(base,'external_knowledge','cache','enterprise-attack.json')]:
  if _ok(p):
   with open(p,'r',encoding='utf-8',errors='ignore') as f:data=json.load(f);break
 else:
  c=os.path.join(base,'external_knowledge','cache');os.makedirs(c,exist_ok=True);p=os.path.join(c,'enterprise-attack.json')
  try:
   with urllib.request.urlopen(ATTACK_URL,timeout=30) as r:
    with open(p,'wb') as f:f.write(r.read())
  except:
   with open(p,'w',encoding='utf-8') as f:json.dump({'objects':[]},f)
  with open(p,'r',encoding='utf-8',errors='ignore') as f:data=json.load(f)
 tm,am,m2t={},{},{}
 for o in data.get('objects',[]):
  if o.get('type')=='attack-pattern':
   tid=''
   for i in o.get('external_references',[]):
    if i.get('source_name')=='mitre-attack' and i.get('external_id','').startswith('T'): tid=i['external_id'];break
   if tid:
    tm[tid]=o.get('name',tid);ph=[p.get('phase_name','').replace('-',' ').strip().title() for p in o.get('kill_chain_phases',[]) if p.get('phase_name')]
    if ph:m2t[tid]=ph
  elif o.get('type')=='x-mitre-tactic':
   xid=''
   for i in o.get('external_references',[]):
    if i.get('source_name')=='mitre-attack' and i.get('external_id','').startswith('TA'): xid=i['external_id'];break
   if xid:am[xid]=o.get('name',xid)
 return tm,am,m2t
def _text(p):
 if p.lower().endswith('.txt'):
  with open(p,'r',encoding='utf-8',errors='ignore') as f:return f.read()
 if p.lower().endswith('.pdf'):
  if pdfplumber is not None:
   try:
    with pdfplumber.open(p) as pdf:
     t='\n'.join((pg.extract_text() or '') for pg in pdf.pages)
     if t.strip(): return t
   except: pass
  if PdfReader is not None:
   try:return '\n'.join((pg.extract_text() or '') for pg in PdfReader(p).pages)
   except: pass
 return ''
def _label(n):
 s=re.sub(r'\s+','',os.path.splitext(n)[0])
 if not s:return ''
 # Use threat actor family prefix as class label (e.g. APT28_xxx -> APT28).
 m=re.match(r'^([A-Za-z][A-Za-z0-9]*?)(?:[_-].*)?$',s)
 return m.group(1).upper() if m else s.upper()
def _safe_title(name,label=''):
 title=os.path.splitext(name)[0]
 # Strip leading dataset label prefix like "APT28_" / "LAZARUS-".
 if label:
  title=re.sub(rf'(?i)^{re.escape(label)}[_ -]+','',title)
 title=re.sub(r'^[A-Za-z0-9 ]+[_ -]+','',title)
 # Remove remaining obvious actor markers to avoid label leakage via filename.
 title=re.sub(r'(?i)\\bapt\\s*[-_]*\\s*\\d+\\b',' ',title)
 title=re.sub(r'(?i)\\b(?:lazarus|turla|oilrig|rocket\\s*kitten|deep\\s*panda|menu\\s*pass|winnti|fin\\s*\\d+)\\b',' ',title)
 title=re.sub(r'[_]+','-',title)
 title=re.sub(r'\\s+',' ',title).strip()
 return title
def _title_entities(name,label=''):
 title=_safe_title(name,label).lower()
 title=re.sub(r'\\b(?:report|analysis|technical|threat|intelligence|campaign|malware|backdoor|trojan|apt|group|operation|unit42|wp|rpt|final|pdf|txt)\\b',' ',title)
 toks=re.findall(r'\\b[a-z][a-z0-9-]{3,}\\b',title)
 bad={'targets','targeting','against','using','update','updated','new','returns','activity','activities','attacks','attack','cyber','espionage','security','research','paper','look','into','under','hood','goes','mobile','more','than','just','game','part','fast','facts'}
 ents=[]
 for tok in toks:
  parts=[p for p in tok.split('-') if len(p)>=3 and p not in bad]
  if len(parts)>=2:
   ents.append('-'.join(parts[:3]))
  elif parts:
   ents.append(parts[0])
 return set(x for x in ents if len(x)>=4 and not x.isdigit())
def _dig(s,d):
 h=hashlib.sha1(str(s).encode('utf-8',errors='ignore')).hexdigest();return torch.tensor([float(int(c,16)%10)/9.0 for c in h[:d]],dtype=torch.float32)
def _style(t,d):
 t=t or '';L=sum(c.isalpha() for c in t);D=sum(c.isdigit() for c in t);S=sum(c.isspace() for c in t);U=sum(c.isupper() for c in t);N=max(1,len(t));WDS=re.findall(r'\b\w+\b',t)
 return _pad(torch.tensor([len(t)/5000.0,L/N,D/N,S/N,U/max(1,L),len(WDS)/1000.0,len(set(w.lower() for w in WDS))/max(1,len(WDS)),max(1,t.count('\n')+1)/200.0],dtype=torch.float32),d)
def _qual(i):
 c=torch.tensor([float(len(i.get(tp,set()))) for tp in IOC_TYPES],dtype=torch.float32);p=torch.tensor([1.0 if len(i.get(tp,set())) else 0.0 for tp in IOC_TYPES],dtype=torch.float32);tot=max(1.0,c.sum().item());return torch.cat([c[:6]/tot,p[:5]])
def _iocs(t,m2t,name='',label=''):
 raw=(t or '');o={tp:set() for tp in IOC_TYPES}
 o['Email'].update(re.findall(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b",raw));o['URL'].update(re.findall(r"https?://[^\s\]\[\)\(<>\"']+",raw));o['Domain'].update(re.findall(r"\b(?:[a-zA-Z0-9-]+\.)+[A-Za-z]{2,}\b",raw));o['IP'].update(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b",raw));o['Vulnerability'].update(x.upper() for x in re.findall(r"\bCVE-\d{4}-\d{4,7}\b",raw,re.I));o['Technique'].update(x.upper() for x in re.findall(r"\bT\d{4}(?:\.\d{3})?\b",raw,re.I));o['Tactic'].update(x for tech in o['Technique'] for x in m2t.get(tech,[]));o['Registry'].update(re.findall(r"\b(?:HKLM|HKCU|HKEY_LOCAL_MACHINE|HKEY_CURRENT_USER)\\[^\n\r\"]+",raw,re.I));o['FilePath'].update(re.findall(r"\b[A-Za-z]:\\[^\n\r:*?\"<>|]{3,}",raw));o['FileName'].update(re.findall(r"\b[a-zA-Z0-9_\-]+\.(?:exe|dll|sys|bat|ps1|vbs|js|hta|docm|xlsm|zip|rar|pdf)\b",raw,re.I));o['Malware'].update(re.findall(r"\b(?:malware|trojan|backdoor|ransomware|implant|tool|loader|rootkit|dropper|rat|wiper)\s+([A-Za-z0-9._-]{3,})",raw,re.I))
 o['Malware'].update(_title_entities(name,label))
 o['Domain']={x for x in o['Domain'] if not x.lower().startswith('http')};return o
def _clean_iocs(i):
 out={tp:set(vs) for tp,vs in i.items()}
 ip=set()
 for x in out['IP']:
  try:
   v=ipaddress.ip_address(x)
   if isinstance(v,ipaddress.IPv4Address) and not (v.is_private or v.is_loopback or v.is_multicast or v.is_reserved or v.is_link_local): ip.add(str(v))
  except ValueError: pass
 out['IP']=ip
 out['Domain']={x.lower().strip('.').strip() for x in out['Domain'] if len(x)>=4 and not x.lower().endswith(NOISY_DOMAIN_SUFFIX)}
 out['URL']={x.strip().rstrip('.,;!?)').lower() for x in out['URL'] if len(x)>10 and ' ' not in x}
 out['Email']={x.lower() for x in out['Email'] if not x.lower().endswith(('@gmail.com','@outlook.com','@hotmail.com'))}
 noisy_files={'setup.exe','install.exe','update.exe','readme.txt','document.pdf','sample.exe','payload.exe','svchost.exe','explorer.exe'}
 out['FileName']={x.lower() for x in out['FileName'] if x.lower() not in noisy_files and len(x)>=5}
 out['FilePath']={x for x in out['FilePath'] if len(x)<180 and not x.lower().startswith(('c:\\windows\\system32','c:\\windows\\syswow64'))}
 out['Registry']={x[:180] for x in out['Registry'] if len(x)>8}
 out['Malware']={x.lower() for x in out['Malware'] if len(x)>=3 and x.lower() not in {'the','and','for','this','that'}}
 return out
def _drop_cross_class_noise(ext,_labels):
 n=max(1,len(ext));df={tp:Counter() for tp in IOC_TYPES}
 for i in ext:
  for tp in IOC_TYPES:
   df[tp].update(i.get(tp,set()))
 max_df={'Registry':3,'FilePath':3,'FileName':4,'Email':3,'URL':3,'Domain':3,'IP':3,'Tactic':max(8,int(0.20*n)),'Technique':max(12,int(0.16*n)),'Vulnerability':6,'Malware':6}
 out=[];removed=Counter()
 for i in ext:
  ni={}
  for tp in IOC_TYPES:
   lim=max_df.get(tp,max(5,int(0.2*n)))
   vals={v for v in i.get(tp,set()) if df[tp][v] <= lim}
   removed[tp]+=len(i.get(tp,set()))-len(vals);ni[tp]=vals
  out.append(ni)
 print('[data_loader] df noise removed: '+', '.join(f'{k}:{v}' for k,v in removed.items() if v))
 return out

def _cap_iocs_for_graph(ext):
 df={tp:Counter() for tp in IOC_TYPES}
 for i in ext:
  for tp in IOC_TYPES:df[tp].update(i.get(tp,set()))
 cap={'Technique':30,'Tactic':3,'Malware':20,'Vulnerability':20,'FileName':22,'Domain':16,'IP':16,'URL':12,'Email':10,'Registry':10,'FilePath':10}
 out=[]
 for i in ext:
  ni={}
  for tp in IOC_TYPES:
   vals=list(i.get(tp,set()))
   vals=sorted(vals,key=lambda v:(df[tp][v],len(str(v)),str(v)))[:cap.get(tp,20)]
   ni[tp]=set(vals)
  out.append(ni)
 return out

def _keywords(t,limit=80):
 toks=re.findall(r'\b[A-Za-z][A-Za-z0-9_\-]{3,}\b',t or '')
 stop={'this','that','with','from','have','were','been','also','their','there','which','using','used','attack','malware','report','analysis','security','system','windows'}
 c=Counter(x.lower() for x in toks if x.lower() not in stop and not x.isdigit())
 return [w for w,_ in c.most_common(limit)]

def _rtext(t,i,n,label=''):
 evidence=[f'Title: {_safe_title(n,label)}']
 priority=['Malware','Technique','Tactic','Vulnerability','FileName','Domain','IP','URL','Email','Registry','FilePath']
 for tp in priority:
  vals=sorted(i.get(tp,set()))[:40 if tp in ('Technique','Tactic','Malware','Vulnerability','FileName') else 20]
  if vals:evidence.append(f'{tp}: '+', '.join(vals))
 kws=_keywords(t,80)
 if kws:evidence.append('Keywords: '+', '.join(kws))
 body=re.sub(r'\s+',' ',t or '').strip()
 if body:
  head=body[:1500];tail=body[-900:] if len(body)>3000 else ''
  evidence.append('Text: '+head)
  if tail:evidence.append('Tail: '+tail)
 return '\n'.join(evidence)
def _ntext(tp,v,tm,am):
 return f'Technique {v}: {tm.get(v,v)}' if tp=='Technique' else (f'Tactic {v}: {am.get(v,v)}' if tp=='Tactic' else f'{tp}: {v}')
def _attr(tp,v,tm):
 x=torch.zeros(64,dtype=torch.float32);x[:len(IOC_TYPES)]=torch.tensor([1.0 if t==tp else 0.0 for t in IOC_TYPES],dtype=torch.float32);x[len(IOC_TYPES):len(IOC_TYPES)+32]=_dig(v,32)
 if tp=='Technique' and v in tm:x[-1]=1.0
 return x
def _hf(base):
 os.environ.setdefault('HF_ENDPOINT',HF_ENDPOINT);p=os.path.join(base,'external_knowledge','hf_models',HF_MODEL)
 if os.path.isdir(p): return AutoTokenizer.from_pretrained(p),AutoModel.from_pretrained(p)
 c=os.path.join(base,'external_knowledge','hf_models')
 try: return AutoTokenizer.from_pretrained(HF_MODEL,cache_dir=c),AutoModel.from_pretrained(HF_MODEL,cache_dir=c)
 except Exception:
  return AutoTokenizer.from_pretrained(HF_MODEL,cache_dir=c,local_files_only=True),AutoModel.from_pretrained(HF_MODEL,cache_dir=c,local_files_only=True)
def _cache_dir(cache,base):
 candidates=[]
 env=os.environ.get('APT_CACHE_DIR','').strip()
 if env:candidates.append(env)
 local_cache=os.path.join(base,'APT-MMF','.cache')
 candidates += [local_cache,cache,os.path.join(tempfile.gettempdir(),'apt_mmf_cache'),r'D:\apt_mmf_cache',r'C:\apt_mmf_cache']
 for c in candidates:
  try:
   os.makedirs(c,exist_ok=True);test=os.path.join(c,'.write_test')
   with open(test,'w',encoding='utf-8') as f:f.write('ok')
   os.remove(test);print(f'[data_loader] cache dir: {c}');return c
  except Exception: pass
 return cache

def _nlt(texts,cache,base,device='cpu'):
 cache_dir=_cache_dir(cache,base);hs=hashlib.sha1(FEATURE_CACHE_VERSION.encode('utf-8'));hs.update(HF_MODEL.encode('utf-8'));hs.update(str(len(texts)).encode('utf-8'))
 for t in texts:
  b=(t or '').encode('utf-8',errors='ignore');hs.update(len(b).to_bytes(8,'little'));hs.update(b)
 h=hs.hexdigest()
 p=os.path.join(cache_dir,f'nlt_feat_{h[:16]}.pt')
 legacy=os.path.join(cache_dir,'nlt_feat.pt')
 if (not _ok(p)) and _ok(legacy):
  try:
   c=torch.load(legacy,map_location='cpu')
   if isinstance(c,dict) and c.get('hash')==h and 'nlt_feat' in c:
    print(f'[data_loader] NLT legacy cache hit: {legacy}')
    try:
     os.makedirs(os.path.dirname(p),exist_ok=True);torch.save(c,p);print(f'[data_loader] NLT cache migrated: {p}')
    except Exception: pass
    return c['nlt_feat']
  except Exception: pass
 if _ok(p):
  try:
   c=torch.load(p,map_location='cpu')
   if isinstance(c,dict) and c.get('hash')==h and 'nlt_feat' in c:
    print(f'[data_loader] NLT cache hit: {p}');return c['nlt_feat']
   print('[data_loader] NLT cache stale; recomputing text features.')
  except Exception: print('[data_loader] NLT cache unreadable; recomputing text features.')
 else: print('[data_loader] NLT cache missing; computing text features.')
 tok,mdl=_hf(base);mdl.to(device);mdl.eval();xs=[]
 with torch.no_grad():
  for i in range(0,len(texts),8):
   e=tok(texts[i:i+8],padding=True,truncation=True,max_length=384,return_tensors='pt')
   e={k:v.to(device) for k,v in e.items()}
   out=mdl(**e).last_hidden_state
   mask=e.get('attention_mask')
   if mask is None:
    pooled=out.mean(1)
   else:
    mask=mask.to(out.dtype).unsqueeze(-1)
    pooled=(out*mask).sum(1)/mask.sum(1).clamp(min=1.0)
   xs.append(pooled.cpu())
 x=torch.cat(xs,0)
 try:
  os.makedirs(os.path.dirname(p),exist_ok=True);torch.save({'hash':h,'version':FEATURE_CACHE_VERSION,'nlt_feat':x},p);print(f'[data_loader] NLT cache saved: {p}')
 except Exception as e: print(f'[data_loader] NLT cache save skipped: {e}')
 return x
def _topo(ntv,ext,idx,nr,d=128):
 n=ntv.shape[0];A=torch.eye(n,dtype=torch.float32)
 for r,i in enumerate(ext):
  for tp in IOC_TYPES:
   for v in i.get(tp,set()):
    nid=idx.get((tp,v))
    if nid is not None:A[r,nid]=A[nid,r]=W.get(tp,1.0)
 deg=A.sum(1);inv=torch.pow(torch.clamp(deg,min=1e-6),-0.5);norm=(inv.unsqueeze(1)*A)*inv.unsqueeze(0);cur=ntv.float();fs=[cur]
 for _ in range(3): cur=norm@cur;fs.append(cur)
 rep=A[:nr].sum(0);return _pad(torch.cat(fs+[torch.stack([deg/torch.clamp(deg.max(),min=1.0),rep/torch.clamp(rep.max(),min=1.0)],1)],1),d)
def _neigh(sets):
 mp={};out=[]
 for i,vals in enumerate(sets):
  for v in vals: mp.setdefault(v,set()).add(i)
 for i,vals in enumerate(sets):
  s=set();[s.update(mp.get(v,set())) for v in vals];s.discard(i);out.append(s)
 return out
def _graph(neigh):
 s=[];d=[]
 for i,vals in enumerate(neigh):
  s.append(i);d.append(i)
  for j in sorted(vals):
   if j!=i:s.append(i);d.append(j)
 return dgl.graph((s,d),num_nodes=len(neigh))
def _mps(ext):
 mp_types=['Technique','Malware','Vulnerability','FileName','Domain','IP','Registry','FilePath','Email','URL']
 if INCLUDE_TACTIC_MP:
  mp_types.insert(1,'Tactic')
 gs=[_graph(_neigh([set(i.get(tp,set())) for i in ext])) for tp in mp_types]
 while len(gs)<20: gs.append(_graph([set() for _ in range(len(ext))]))
 return gs[:20]
def _segments(t):
 if len(t)<SEGMENT_MIN_CHARS:return [(t,0)]
 starts=[];pos=0
 while pos<len(t) and len(starts)<MAX_SEGMENTS_PER_REPORT:
  starts.append(pos);pos+=max(1,SEGMENT_STRIDE)
 if starts and starts[-1]+SEGMENT_CHARS<len(t) and len(starts)<MAX_SEGMENTS_PER_REPORT: starts.append(max(0,len(t)-SEGMENT_CHARS))
 out=[]
 for k,st in enumerate(starts[:MAX_SEGMENTS_PER_REPORT]):
  seg=t[st:st+SEGMENT_CHARS]
  if seg.strip():out.append((seg,k))
 return out or [(t,0)]

def _print_stats(rlab,groups,tr,va,te,split_seed=None):
 seed_used=SPLIT_SEED if split_seed is None else split_seed
 print(f'[data_loader] reports={len(set(groups))}, samples={len(rlab)}, classes={len(set(rlab))}, split_seed={seed_used}, segment_aug={ENABLE_SEGMENT_AUG}')
 print('[data_loader] class distribution: '+', '.join(f'{k}:{v}' for k,v in sorted(Counter(rlab).items())))
 for name,ids in [('train',tr),('val',va),('test',te)]:
  c=Counter([rlab[i] for i in ids]);print(f'[data_loader] {name} size={len(ids)}, groups={len(set(groups[i] for i in ids))}: '+', '.join(f'{k}:{v}' for k,v in sorted(c.items())))

def _split(labels,groups=None, seed=SPLIT_SEED):
 if groups is None: groups=list(range(len(labels)))
 y=np.array(labels);groups=np.array(groups);rng=np.random.RandomState(seed);tr=[];va=[];te=[]
 group_label={g:y[np.where(groups==g)[0][0]] for g in sorted(set(groups.tolist()))}
 by_cls=defaultdict(list)
 for g,lab in group_label.items(): by_cls[lab].append(g)
 for c in sorted(by_cls):
  gs=by_cls[c];rng.shuffle(gs);m=len(gs)
  if m==1: tr+=gs
  elif m==2: tr+=gs[:1];te+=gs[1:]
  else:
   nva=max(1,int(round(m*0.1)));nte=max(1,int(round(m*0.1)));ntr=max(1,m-nva-nte)
   while ntr+nva+nte>m:
    if ntr>=nva and ntr>1:ntr-=1
    elif nva>=nte and nva>1:nva-=1
    else:nte-=1
   tr+=gs[:ntr];va+=gs[ntr:ntr+nva];te+=gs[ntr+nva:ntr+nva+nte]
 def expand(gs):
  ids=[]
  for g in gs: ids+=np.where(groups==g)[0].tolist()
  ids=np.array(ids,dtype=np.int64);rng.shuffle(ids);return ids
 return expand(tr),expand(va),expand(te)
def load_cti_kg(seed=SPLIT_SEED):
 base=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'));dd=_dataset(base);cache=os.path.join(base,'external_knowledge','cache');tm,am,m2t=_attack(base)
 paths=[p for p in sorted(glob.glob(os.path.join(dd,'*'))) if os.path.isfile(p) and p.lower().endswith(('.txt','.pdf'))]
 rtxts=[];rlab=[];ext=[];qual=[];sty=[];raw_texts=[];raw_names=[];groups=[]
 for p in paths:
  n=os.path.basename(p);lab=_label(n)
  if not lab: continue
  t=_text(p)
  segs=_segments(t) if ENABLE_SEGMENT_AUG else [(t,0)]
  for seg,si in segs:
   if len(re.sub(r'\s+','',seg or '')) < 200:
    continue
   i=_clean_iocs(_iocs(seg,m2t,n,lab));sv=_style(seg,20)
   if sum(len(i.get(tp,set())) for tp in IOC_TYPES)==0 and len(seg)<800:
    continue
   rlab.append(lab);ext.append(i);qual.append(_qual(i));sty.append(sv);raw_texts.append(seg);raw_names.append(n if si==0 else f'{n}#seg{si+1}');groups.append(n)
 if not rlab: raise RuntimeError(f'No report files found in dataset directory: {dd}')
 text_ext=_drop_cross_class_noise(ext,rlab)
 rtxts=[_rtext(t,i,n,lab) for t,i,n,lab in zip(raw_texts,text_ext,raw_names,rlab)]
 ext=_cap_iocs_for_graph(text_ext)
 mp={x:i for i,x in enumerate(sorted(set(rlab)))};labels=torch.tensor([mp[x] for x in rlab],dtype=torch.long);num_classes=len(mp)
 nts=['APT_Report']*len(rtxts);ntxt=list(rtxts);idx={}
 for i in ext:
  for tp in IOC_TYPES:
   for v in sorted(i.get(tp,set()),key=str):
    if (tp,v) not in idx: idx[(tp,v)]=len(nts);nts.append(tp);ntxt.append(_ntext(tp,v,tm,am))
 nr=len(rtxts);nn=len(nts);heter=torch.zeros((nr,nn),dtype=torch.float32);report=torch.zeros((nr,nn),dtype=torch.float32)
 for r,i in enumerate(ext):
  report[r,r]=1.0
  for tp in IOC_TYPES:
   for v in i.get(tp,set()):
    nid=idx.get((tp,v))
    if nid is not None: heter[r,nid]=W.get(tp,1.0)
 t2i={t:i for i,t in enumerate(NODE_TYPE)};ntv=torch.zeros((nn,len(NODE_TYPE)),dtype=torch.float32)
 for i,tp in enumerate(nts): ntv[i,t2i.get(tp,t2i['APT_Report'])]=1.0
 rev={v:k for k,v in idx.items()};rows=[]
 for i in range(nn):
  if i<nr:
   row=torch.zeros(64,dtype=torch.float32);row[:11]=qual[i];row[-20:]=sty[i];rows.append(row)
  else:
   tp,v=rev[i];rows.append(_attr(tp,v,tm))
 attr=torch.stack(rows,0);nlt=_nlt(ntxt,cache,base,device='cuda:0' if torch.cuda.is_available() else 'cpu');topo=_topo(ntv,ext,idx,nr,128);mps=_mps(ext)
 print(f'[data_loader] graph nodes={nn}, reports={nr}, ioc_nodes={nn-nr}, heter_edges={(heter>0).sum().item()}')
 train_idx,val_idx,test_idx=_split(rlab,groups,seed)
 _print_stats(rlab,groups,train_idx,val_idx,test_idx,split_seed=seed)
 train_mask=torch.zeros(nr,dtype=torch.bool);val_mask=torch.zeros(nr,dtype=torch.bool);test_mask=torch.zeros(nr,dtype=torch.bool)
 train_mask[train_idx]=True;val_mask[val_idx]=True;test_mask[test_idx]=True
 return labels,num_classes,train_mask,val_mask,test_mask,attr,nlt,topo,ntv,heter,report,mps
