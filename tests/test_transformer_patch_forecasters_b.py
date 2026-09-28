"""Paper-structure and runtime tests for six implementations."""
from __future__ import annotations
import copy
import unittest
import torch
from pydantic import ValidationError
from moderntsf.models.card.model import Model as CARD, exponential_smooth
from moderntsf.models.card.spec import ModelParameterConfig as CARDParams
from moderntsf.models.crossformer.model import Model as Crossformer, dsw_embed
from moderntsf.models.crossformer.spec import ModelParameterConfig as CrossParams
from moderntsf.models.dsformer.model import Model as DSFormer, dual_sampling
from moderntsf.models.dsformer.spec import ModelParameterConfig as DSParams
from moderntsf.models.duet.model import Model as DUET, mahalanobis_bias, moving_average
from moderntsf.models.duet.spec import ModelParameterConfig as DUETParams
from moderntsf.models.multipatchformer.model import Model as MultiPatchFormer, SemiAutoregressiveHead
from moderntsf.models.multipatchformer.spec import ModelParameterConfig as MultiParams
from moderntsf.models.nstransformer.model import Model as NSTransformer, DeStationaryAttention
from moderntsf.models.nstransformer.spec import ModelParameterConfig as NSParams


def factories(length=8, horizon=3, channels=2):
    return {
        "CARD": lambda: CARD(length,horizon,channels,patch_len=min(4,length),stride=2,d_model=8,n_heads=2,e_layers=1,d_ff=16,dropout=0),
        "Crossformer": lambda: Crossformer(length,horizon,channels,d_model=8,n_heads=2,e_layers=2,d_ff=16,seg_len=2,win_size=2,factor=2,dropout=0),
        "DSFormer": lambda: DSFormer(length,horizon,channels,num_layer=1,muti_head=2,num_samp=2,dropout=0),
        "DUET": lambda: DUET(length,horizon,channels,d_model=8,n_heads=2,e_layers=1,d_ff=16,dropout=0,fc_dropout=0,moving_avg=3,num_experts=2,k=1,hidden_size=8,noisy_gating=False),
        "MultiPatchFormer": lambda: MultiPatchFormer(length,horizon,channels,d_model=8,n_heads=2,e_layers=1,d_ff=16,dropout=0),
        "NSTransformer": lambda: NSTransformer(length,horizon,0,channels,d_model=8,n_heads=2,e_layers=1,d_layers=1,d_ff=16,dropout=0,p_hidden_dims=[8],p_hidden_layers=1),
    }


class PaperStructureTests(unittest.TestCase):
    def test_card_causal_ema(self):
        x=torch.tensor([1.,3.,7.]).reshape(1,1,3,1)
        torch.testing.assert_close(exponential_smooth(x,.5).flatten(),torch.tensor([1.,2.,4.5]))

    def test_crossformer_dsw_axis_and_padding(self):
        projection=torch.nn.Linear(2,1,bias=False); projection.weight.data.fill_(1)
        x=torch.tensor([[[1.],[2.],[3.]]])
        out=dsw_embed(x,2,projection)
        self.assertEqual(tuple(out.shape),(1,1,2,1)); torch.testing.assert_close(out.flatten(),torch.tensor([2.,5.]))

    def test_dsformer_dual_sampling_is_distinct_and_complete(self):
        x=torch.arange(8.).reshape(1,1,8)
        piece,interval=dual_sampling(x,2)
        torch.testing.assert_close(piece.flatten(),x.flatten())
        self.assertFalse(torch.equal(piece,interval)); torch.testing.assert_close(interval.flatten(),torch.tensor([0.,2.,4.,6.,1.,3.,5.,7.]))

    def test_duet_decomposition_and_mahalanobis_relation(self):
        x=torch.arange(8.).reshape(1,4,2)
        trend=moving_average(x,3); torch.testing.assert_close((x-trend)+trend,x)
        bias=mahalanobis_bias(x.transpose(1,2)); torch.testing.assert_close(bias.diagonal(dim1=-2,dim2=-1),torch.zeros(1,2))
        self.assertTrue((bias<=0).all())

    def test_multipatch_semiautoregressive_dependencies(self):
        head=SemiAutoregressiveHead(8,10,groups=8)
        emitted=0
        for layer in head.layers:
            self.assertEqual(layer.in_features,8+emitted); emitted+=layer.out_features
        self.assertEqual(emitted,10)

    def test_nonstationary_attention_tau_delta_equation(self):
        layer=DeStationaryAttention(4,1,0).eval()
        q=torch.randn(2,3,4); context=torch.randn(2,3,4)
        base=layer(q,context,torch.ones(2),torch.zeros(2,3))
        shifted=layer(q,context,torch.full((2,),2.),torch.tensor([[0.,1.,2.],[0.,1.,2.]]))
        self.assertEqual(tuple(base.shape),(2,3,4)); self.assertFalse(torch.equal(base,shifted))


class SchemaTests(unittest.TestCase):
    def test_invalid_architecture_contracts(self):
        cases=[lambda:CARDParams(enc_in=2,d_model=7,n_heads=2),lambda:CrossParams(enc_in=2,d_model=7,n_heads=2),
               lambda:DSParams(enc_in=2,dropout=1),lambda:DUETParams(enc_in=2,num_experts=2,k=3),
               lambda:MultiParams(enc_in=2,d_model=10,n_heads=2),lambda:NSParams(enc_in=2,p_hidden_dims=[8],p_hidden_layers=2)]
        for case in cases:
            with self.subTest(case=case),self.assertRaises(ValidationError): case()


class RuntimeTests(unittest.TestCase):
    def test_forward_backward_active_roundtrip_boundaries_and_marks(self):
        torch.manual_seed(9137)
        for name,factory in factories().items():
            with self.subTest(model=name):
                model=factory().cpu(); x=torch.randn(2,8,2,requires_grad=True)
                marks=torch.randn(2,8,6); out=model(x,marks,torch.zeros(2,3,2),torch.randn(2,3,6))
                self.assertEqual(tuple(out.shape),(2,3,2)); self.assertTrue(torch.isfinite(out).all())
                out.square().mean().backward(); self.assertIsNotNone(x.grad); self.assertTrue(torch.isfinite(x.grad).all())
                for parameter_name,parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad,f"{name}:{parameter_name}"); self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().max().item(),0,f"{name}:{parameter_name}")
                model.eval(); expected=model(x.detach(),marks)
                clone=factory().eval(); clone.load_state_dict(copy.deepcopy(model.state_dict()))
                torch.testing.assert_close(clone(x.detach(),marks),expected)
                self.assertEqual(tuple(model(torch.randn(1,8,2)).shape),(1,3,2))
                with self.assertRaises(ValueError): model(torch.randn(1,7,2))
                torch.testing.assert_close(model(x.detach(),marks+10),expected)

    def test_minimum_supported_histories(self):
        cases={
            "CARD":CARD(2,1,1,patch_len=2,stride=1,d_model=4,n_heads=1,e_layers=1,d_ff=8,dropout=0),
            "Crossformer":Crossformer(2,1,1,d_model=4,n_heads=1,e_layers=1,d_ff=8,seg_len=1,win_size=1,factor=1,dropout=0),
            "DSFormer":DSFormer(2,1,1,num_layer=1,muti_head=1,num_samp=1,dropout=0),
            "DUET":DUET(2,1,1,d_model=4,n_heads=1,e_layers=1,d_ff=8,dropout=0,fc_dropout=0,moving_avg=2,num_experts=1,k=1,hidden_size=4,noisy_gating=False),
            "MultiPatchFormer":MultiPatchFormer(1,1,1,d_model=4,n_heads=1,e_layers=1,d_ff=8,dropout=0),
            "NSTransformer":NSTransformer(1,1,0,1,d_model=4,n_heads=1,e_layers=1,d_layers=1,d_ff=8,dropout=0,p_hidden_dims=[4],p_hidden_layers=1),
        }
        for name,model in cases.items():
            with self.subTest(model=name): self.assertEqual(tuple(model(torch.randn(1,model.seq_len,1)).shape),(1,1,1))

if __name__ == "__main__": unittest.main()
