import sys,json,hashlib,time
from fractions import Fraction
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from padjective import db,benchmark_runtime as r
from padjective.paper_revision_experiments import _load_paper_dataset
from padjective.paper_published_methods import PAPER_SNAPSHOT,_snapshot_digest
from padjective.paper_randomised_methods import SNAPSHOT_DIGEST
from padjective.paper_low_q_experiment import EXPERIMENT
import argparse
parser=argparse.ArgumentParser(description='Read-only frozen taxonomy-association audit; no database writes.')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
out={'source_commit':'b98d1871934010bd03a5126f52dae02a48604b74','folds':[]}
with db.get_connection() as c:
 c.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
 d=_load_paper_dataset(c,snapshot_ref=PAPER_SNAPSHOT,schema='padjective')
 assert d.prime_base==71 and _snapshot_digest(d)==SNAPSHOT_DIGEST
 archived={f:e for f,e in c.execute("select cv_fold,evidence from padjective.paper_low_q_models where experiment_key=%s and q_label='71' and family='greedy_taxonomy_association' and fraction=1 and member_index=-1",(EXPERIMENT,))}
out.update(snapshot_digest=_snapshot_digest(d),products=len(d.records),archived_folds=len(archived))
def exact_medoid(values,p,candidates=None):
 counts=Counter(map(int,values)); candidates=sorted(set(counts if candidates is None else candidates))
 def cost(c):
  ans=Fraction(0)
  for v,n in counts.items():
   x=abs(c-v)
   if not x:continue
   den=1
   while x%p==0:x//=p;den*=p
   ans+=Fraction(n,den)
  return ans
 costs={c:cost(c) for c in candidates}
 return min(candidates,key=lambda c:(costs[c],c)),costs
def independent_fit(fold,records):
 training=[x for x in records if x.cv_fold!=fold]
 support={}; classes={}
 for i,row in enumerate(training):
  for tag in row.tags:
   support.setdefault(tag,[]).append(i)
   classes.setdefault(tag,Counter())[row.taxonomy_id]+=1
 order=sorted(support,key=lambda tag:(-Fraction(max(classes[tag].values()),len(support[tag])),-len(support[tag]),tag))
 residuals=[row.encoded_path for row in training]; weights={}
 for tag in order:
  w,_=exact_medoid([residuals[i] for i in support[tag]],71)
  weights[tag]=w
  for i in support[tag]:residuals[i]-=w
 totals=[sum(weights[tag] for tag in row.tags) for row in training]
 labels=[row.encoded_path for row in training]
 zero_labels=[y for y,t in zip(labels,totals) if t==0]
 default=exact_medoid(zero_labels,71,labels)[0] if zero_labels else Counter(labels).most_common(1)[0][0]
 pred=[sum(weights.get(tag,0) for tag in row.tags) or default for row in records if row.cv_fold==fold]
 return weights,default,pred
original=r._select_coefficient; original_default=r._select_default_prediction
for fold in range(5):
 check={'fold':fold,'coordinates':0,'nonoptimal':[],'different_exact_ties':[],'defaults':[]}
 def audited(values,p):
  actual=original(values,p); exact,costs=exact_medoid(values,p)
  check['coordinates']+=1
  if actual!=exact:
   key='nonoptimal' if costs[actual]>costs[exact] else 'different_exact_ties'
   check[key].append({'step':check['coordinates'],'actual':actual,'exact':exact,'cost_gap':str(costs[actual]-costs[exact])})
  return actual
 def auditdefault(values,candidates,p):
  actual=original_default(values,candidates,p)
  if values:
   exact,costs=exact_medoid(values,p,candidates)
   check['defaults'].append({'actual':actual,'exact':exact,'cost_gap':str(costs[actual]-costs[exact])})
  return actual
 r._select_coefficient=audited;r._select_default_prediction=auditdefault
 fit=r.umllr_run_fold(fold,d.records,(),71,tag_order_strategy='taxonomy_association')
 pred=[int(x.predicted_value) for x in fit.predictions]
 check['stored_predictions_match']=pred==[int(x) for x in archived[fold]['predictions']]
 check['prediction_sha256']=hashlib.sha256(json.dumps(pred,separators=(',',':')).encode()).hexdigest()
 # Fresh exact fit allows earlier tie changes to propagate through later residuals.
 r._select_coefficient=lambda values,p:exact_medoid(values,p)[0]
 r._select_default_prediction=lambda values,candidates,p:exact_medoid(values,p,candidates)[0] if values else original_default(values,candidates,p)
 fresh=r.umllr_run_fold(fold,d.records,(),71,tag_order_strategy='taxonomy_association')
 ep=[int(x.predicted_value) for x in fresh.predictions]
 iw,ib,ip=independent_fit(fold,d.records)
 assert iw=={x.tag:x.coefficient for x in fresh.coefficients} and ib==fresh.default_prediction and ip==ep
 check['independent_fit_matches_exact_replay']=True
 def score(predictions,q):
  total=Fraction(0)
  targets=[x.encoded_path for x in d.records if x.cv_fold==fold]
  for a,b in zip(targets,predictions):
   delta=abs(a-b);den=1
   if not delta:continue
   while delta%71==0:delta//=71;den*=q
   total+=Fraction(1,den)
  return float(total/len(targets))
 check['original_losses']={str(q):score(pred,q) for q in (2,71)}
 check['exact_losses']={str(q):score(ep,q) for q in (2,71)}
 check['exact_fit_changed_predictions']=sum(a!=b for a,b in zip(pred,ep))
 check['exact_fit_changed_coefficients']=sum(a.coefficient!=b.coefficient for a,b in zip(fit.coefficients,fresh.coefficients))
 out['folds'].append(check)
 print(json.dumps(check),flush=True)
 args.output.write_text(json.dumps(out,indent=2)+'\n')
