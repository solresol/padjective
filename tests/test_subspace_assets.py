"""Audit the saved research evidence and exact-value manuscript-side tables."""
import hashlib
import json
from pathlib import Path
import re

import pytest

from padjective.paper_subspace_analysis import analyse
from padjective.subspace_voting import FRACTIONS

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs/results/subspace-voting-20260911/results.json'
SHA256='0bb9c4cb5141850df87ed8bb9d8e610c2739b03de11db405c3b012158ea8cde9'


def numerical_equal(actual,expected):
    if isinstance(expected,dict):
        assert set(actual)==set(expected)
        for key in expected:
            numerical_equal(actual[key],expected[key])
    elif isinstance(expected,list):
        assert len(actual)==len(expected)
        for a,e in zip(actual,expected):
            numerical_equal(a,e)
    elif isinstance(expected,float):
        assert actual==pytest.approx(expected,rel=1e-11,abs=2e-13)
    else:
        assert actual==expected


def test_complete_saved_analysis_recomputes():
    raw=DATA.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SHA256
    report=json.loads(raw)
    assert report['validation']['status']=='passed'
    assert len(report['components'])==6075 and len(report['fold_results'])==6125
    numerical_equal(analyse(report['fold_results']),report['analysis'])
    assert all(r['exact_correct']==round(r['metrics']['n']*r['metrics']['exact_accuracy'])
               and r['root_correct']==round(r['metrics']['n']*r['metrics']['first_digit_accuracy'])
               for r in report['fold_results'])
    assert all(r['feature_union_count']==r['eligible_features'] for r in report['fold_results'] if r['members']==243)


def test_export_contains_no_private_observation_or_model_vectors():
    prohibited={'predictions','coefficients','training_predictions','training_raw_predictions',
                'held_out_raw_predictions','feature_names','mask','product_id','product_key','title','tags'}
    def walk(value):
        if isinstance(value,dict):
            assert not prohibited.intersection(value)
            for child in value.values():
                walk(child)
        elif isinstance(value,list):
            for child in value:
                walk(child)
    walk(json.loads(DATA.read_text()))


def test_result_note_tables_match_saved_evidence():
    report=json.loads(DATA.read_text())
    text=(ROOT/'docs/subspace-voting-results.md').read_text()
    cells={(c['fraction'],c['members'],c['rule']):c for c in report['analysis']['cells']}
    rows=re.findall(r'^\| (1|3|9|15|27|45|81|135|243) \| (.+) \|$',text,re.MULTILINE)
    assert len(rows)==9
    for size,values in rows:
        assert values.split(' | ')==[f"{cells[f,int(size),'raw_valid_medoid']['mean_metrics']['mean_padic_loss']:.6f}" for f in FRACTIONS]
    rows=re.findall(r'^\| (12\.5|25|50|75|100)% \| (.+) \|$',text,re.MULTILINE)
    assert len(rows)==5
    for percent,values in rows:
        f=float(percent)/100
        metrics=cells[f,243,'raw_valid_medoid']['mean_metrics']
        wanted=[f"{metrics['mean_padic_loss']:.6f}",f"{100*metrics['exact_accuracy']:.3f}%",
                f"{100*metrics['first_digit_accuracy']:.3f}%"]
        wanted.append('Reference' if f==1. else f"{next(c for c in report['analysis']['primary_contrasts'] if c.get('kind')=='subset' and c['fraction']==f)['holm_p_value']:.6f}")
        assert values.split(' | ')==wanted
    section=text.split('## Prespecified primary tests')[1].split('## Representation results')[0]
    rows=[line.split(' | ')[1:] for line in section.splitlines() if line.startswith('| ')][1:]
    assert len(rows)==12
    for row,c in zip(rows,report['analysis']['primary_contrasts']):
        lo,hi=c['bonferroni_simultaneous_ci95']
        assert row==[f"{c['mean_difference']:+.6f}",f"[{lo:+.6f}, {hi:+.6f}]",f"{c['holm_p_value']:.6f} |"]
