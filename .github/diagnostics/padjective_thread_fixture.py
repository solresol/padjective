import sys,pytest
from threadpoolctl import threadpool_limits,threadpool_info
restore=sys.argv[1]=='restored'
class Probe:
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self,item):
        if item.name!='test_notebook_keeps_legacy_environment_override':
            yield
            return
        counts=lambda: [(x['prefix'],x['num_threads']) for x in threadpool_info()]
        print('before notebook:',counts(),flush=True)
        if restore:
            with threadpool_limits():
                yield
                print('after notebook before restoration:',counts(),flush=True)
            print('after restoration:',counts(),flush=True)
        else:
            yield
            print('after notebook without restoration:',counts(),flush=True)
raise SystemExit(pytest.main(['-q','-s',
 'tests/test_product_taxonomy_bench_export.py::test_notebook_keeps_legacy_environment_override',
 'tests/test_taxonomy_levelwise_classifier.py::test_levelwise_classifier_predicts_valid_taxonomy_leaf'],plugins=[Probe()]))
