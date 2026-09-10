import copy

import pytest

from padjective import live_subspace as live
from padjective.product_hash import hash_product_url


def source(i, *, domain='example.myshopify.com', handle=None, tags='Red, Shirt',
           taxonomy='tax1', path='1.1', day=1, url=None):
    return dict(id=i, myshopify_domain=domain, product_handle=handle or f'product-{i}',
        product_url=url, created_at=f'2026-09-{day:02d}', updated_at=f'2026-09-{day:02d}',
        has_title=True, has_details=True, tags=tags, taxonomy_id=taxonomy,
        numeric_path=path, taxonomy_source='gold_llm', classified_at=f'2026-09-{day:02d}')


def test_keep_empty_rare_labels_and_latest_entity_alias_overlap():
    old = source(1, handle='a', url='https://example.com/products/a?x=1')
    latest = source(2, handle='a', day=2, url='https://example.net/products/a', tags='', taxonomy='rare', path='2')
    rows, counts = live.prepare_rows([old, source(3), latest],
        {hash_product_url('https://example.com/products/a')})
    retained = next(r for r in rows if r['source_product_id'] == 2)
    assert retained['tags'] == [] and retained['paper_overlap']
    assert retained['taxonomy_id'] == 'rare'
    assert counts['duplicate_entity_rows'] == 1
    assert counts['entities_with_conflicting_historical_labels'] == 1
    assert counts['retained_products'] == 2 and counts['empty_tags'] == 1


def test_source_exclusions_and_canonical_dedup():
    a = source(1, url='https://example.com/products/a?x=1')
    b = source(2, day=2, url='https://example.com/products/a#top', taxonomy='tax2', path='2')
    bad = source(3)
    bad['taxonomy_source'] = 'padjective_prediction'
    missing = source(4)
    missing['has_details'] = False
    unresolved = source(5, path=None)
    rows, counts = live.prepare_rows([a, bad, missing, unresolved, b], set())
    assert len(rows) == 1 and rows[0]['source_product_id'] == 2
    assert counts['duplicate_canonical_urls'] == 1
    assert counts['duplicate_url_label_conflicts'] == 1
    assert counts['missing_reference_label'] == 1
    assert counts['missing_details'] == 1 and counts['unresolved_reference_path'] == 1


def test_capacity_and_encoding_fail_closed(monkeypatch):
    monkeypatch.setattr(live, 'MAX_ROWS', 1)
    with pytest.raises(ValueError, match='no truncation'):
        live.prepare_rows([source(1), source(2)], set())
    assert live.encode('1.2.3.4.5.6.7.80') < live.P**live.E
    for path in ('1.0.2', '1.83', '1.2.3.4.5.6.7.8.9'):
        with pytest.raises(ValueError):
            live.encode(path)


def test_stable_store_folds_and_training_only_vocabulary():
    assert live.store_fold('Example.MyShopify.com ') == live.store_fold('example.myshopify.com')
    rows, _ = live.prepare_rows([source(i) for i in range(10)], set())
    fold = rows[0]['fold']
    assert all(r['fold'] == fold for r in rows)
    # Manufacture the split to check feature-frequency eligibility independently.
    for i, row in enumerate(rows):
        row['fold'] = 0 if i < 5 else 1
        row['tags'] = ['common', 'test-only'] if i < 5 else ['common', 'train-only']
    c = live.context(rows, 0)
    assert c['vocabulary'] == ['common', 'train-only']
    assert c['x_test'].sum(axis=1).A.ravel().tolist() == [1]*5
    assert c['x_train'].sum(axis=1).A.ravel().tolist() == [2]*5


def test_trend_separates_changed_new_removed_and_common():
    old, _ = live.prepare_rows([source(1), source(2), source(3)], set())
    new = copy.deepcopy(old[:2])
    new[1]['content_sha256'] = 'changed'
    extra, _ = live.prepare_rows([source(99)], set())
    new += extra
    before = {r['product_key']: {'0.75': r['target'], '1.0': r['target']} for r in old}
    after = {r['product_key']: {'0.75': r['target'], '1.0': r['target']} for r in new}
    trend = live.compare_trend(old, new, before, after)
    assert trend['unchanged_common'] == trend['changed_common'] == 1
    assert trend['new_products'] == trend['removed_products'] == 1
    assert trend['0.75']['current']['mean_padic_loss'] == 0


def test_source_query_uses_reference_source_and_unique_crawl_join():
    assert "pt.taxonomy_source='gold_llm'" in live.SOURCE_SQL
    assert 'USING(myshopify_domain,run_name,product_handle)' in live.SOURCE_SQL
    assert 'umllr_predictions' not in live.SOURCE_SQL
