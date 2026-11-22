import numpy as np
import pytest
from experiments.ms_swift_aqa.papers28.paired import check_inference_config

from experiments.ms_swift_aqa.papers28.paired import paired


def test_bootstrap_uses_macro_recall_and_paired_clusters():
    labels = np.array([0,0,0,1,1,0,1])
    clusters = ['a','a','a','b','b','c','c']
    a = np.array([1,1,1,0,0,1,0],dtype=bool)
    b = np.array([0,0,0,1,1,0,1],dtype=bool)
    result = paired({'m':a,'lora':b,'frozen-base':a},clusters,labels,3,1000)
    d = result['contrasts']['m vs lora']
    assert d['delta_pp'] == pytest.approx(0.)
    assert d['candidate_only_correct'] == 4 and d['reference_only_correct'] == 3
    assert result['clusters'] == 3 and result['valid_draws'] < 1000
    assert result['contrasts']['m vs frozen-base']['percentile_95_pp'] == [0.,0.]
    # The same predictions on a single accuracy scale have a different estimand.
    acc = paired({'m':a,'lora':b},clusters,np.zeros(7),3,1000)
    assert acc['contrasts']['m vs lora']['delta_pp'] == pytest.approx(100/7)


def test_same_weight_with_different_adapter_config_is_rejected(tmp_path):
    selected = {'method':'lora','config_sha256':'a','files':{'a':'1'}}
    check_inference_config({'identity':{'adapter_config_sha256':'a'}},selected,tmp_path,False)
    with pytest.raises(ValueError,match='configuration'):
        check_inference_config({'config_sha256':'b'},selected,tmp_path,False)
    with pytest.raises(ValueError,match='sidecar'):
        check_inference_config({'config_sha256':'a','files':{'a':'2'}},selected,tmp_path,False)
