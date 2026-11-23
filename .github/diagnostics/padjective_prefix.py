from pathlib import Path
import json,subprocess,sys,os
TARGET='tests/test_taxonomy_levelwise_classifier.py::test_levelwise_classifier_predicts_valid_taxonomy_leaf'
META=Path(os.environ['RUNNER_TEMP'])/'padjective-prefix.json'
def collect_prefix(n):
    import pytest
    class Prefix:
        @pytest.hookimpl(trylast=True)
        def pytest_collection_modifyitems(self,config,items):
            idx=next(i for i,q in enumerate(items) if q.nodeid==TARGET)
            if n<0: keep=[]
            else: keep=items[:min(idx,n)]+[items[idx]]
            META.write_text(json.dumps({'preceding':idx,'last':keep[-2].nodeid if len(keep)>1 else None}))
            gone=[q for q in items if q not in keep]
            config.hook.pytest_deselected(items=gone)
            items[:]=keep
    return pytest.main(['-q','-m','not slow']+(['--collect-only'] if n<0 else []),plugins=[Prefix()])
def driver():
    def run(n):
        p=subprocess.run([sys.executable,__file__,str(n)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=100)
        meta=json.loads(META.read_text())
        print(f"prefix={n} rc={p.returncode} last={meta['last']}",flush=True)
        if p.returncode not in (0,5):
            for l in p.stdout.splitlines():
                if any(w in l for w in ('Segmentation fault','train_levelwise_models','_lbfgsb_py.py','failed,','ERROR')):
                    print(l,flush=True)
        return p.returncode,meta['preceding']
    a,_=run(253)
    b,_=run(254)
    print(f'Exact hosted prefix boundary 253({a}) -> 254({b})',flush=True)
if len(sys.argv)==1:driver()
else:raise SystemExit(collect_prefix(int(sys.argv[1])))
