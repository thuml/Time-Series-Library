import torch

from official_decomposition_label_experiment import configuration
from models.PatchTST import Model


def test_switch_builds_official_two_branches_and_preserves_target_shape():
    cfg=dict(seq_len=32,pred_len=8,e_layers=1,n_heads=2,d_model=8,d_ff=16,
             dropout=0.,patch_len=8,stride=4,kernel_size=5)
    torch.manual_seed(2021)
    opened=Model(configuration(cfg,2,True))
    torch.manual_seed(2021)
    closed=Model(configuration(cfg,2,False))
    assert opened.decomposition is True and closed.decomposition is False
    assert hasattr(opened,'model_trend') and hasattr(opened,'model_res')
    assert sum(p.numel() for p in opened.parameters())==2*sum(p.numel() for p in closed.parameters())
    for key,value in closed.model.state_dict().items():
        torch.testing.assert_close(opened.model_trend.state_dict()[key],value,rtol=0,atol=0)
    x=torch.randn(2,32,2)
    residual,trend=opened.decomp_module(x)
    torch.testing.assert_close(residual+trend,x)
    prediction=opened(x)
    assert prediction.shape==(2,8,2) and torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    for branch in [opened.model_trend,opened.model_res]:
        assert branch.backbone.W_P.weight.grad.abs().max()>0
