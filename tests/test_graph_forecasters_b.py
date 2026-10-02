"""Structure and complete runtime tests for graph implementation batch B."""
from __future__ import annotations
import copy
import unittest
import numpy as np
import torch
from tsflab.models.bigst.model import Model as BigST
from tsflab.models.gagnn.model import Model as GAGNN
from tsflab.models.mtgnn.model import Model as MTGNN
from tsflab.models.megacrn.model import Model as MegaCRN
from tsflab.models.pm25gnn.model import Model as PM25GNN
from tsflab.models.stgode.model import Model as STGODE

def graph(nodes=4):
    value=np.eye(nodes,dtype=np.float32)
    for i in range(nodes-1): value[i,i+1]=1; value[i+1,i]=.5
    return value
def marks(batch,steps,offset=0):
    rows=[[2026,8,1+i//24,5,(i+offset)%24,0] for i in range(steps)]
    return torch.tensor([rows]*batch,dtype=torch.float32)
def factories(length=6,horizon=3,nodes=4,adj=None):
    adj=graph(nodes) if adj is None else adj
    return {
      "BigST":lambda:BigST(length,horizon,nodes,adj,input_dim=3,hid_dim=8,node_dim=4,time_dim=4,random_feature_dim=6,dropout=0,use_bn=False),
      "GAGNN":lambda:GAGNN(length,horizon,nodes,adj,cov_dim=2,d_model=8,num_layers=2,dropout=0,group_num=2),
      "MTGNN":lambda:MTGNN(length,horizon,nodes,adj,input_dim=3,gcn_depth=1,subgraph_size=nodes,node_dim=4,conv_channels=8,residual_channels=8,skip_channels=8,end_channels=8,layers=2,dropout=0),
      "MegaCRN":lambda:MegaCRN(length,horizon,nodes,adj,input_dim=3,rnn_units=8,cheb_k=2,mem_num=4,mem_dim=4),
      "PM25_GNN":lambda:PM25GNN(length,horizon,nodes,adj,cov_dim=2,hid_dim=8),
      "STGODE":lambda:STGODE(length,horizon,nodes,adj,input_dim=3,hidden_dim=8,ode_steps=1),
    }

class PaperStructureTests(unittest.TestCase):
    def test_bigst_positive_linear_attention(self):
        model=factories()["BigST"](); features=model._positive_features(torch.randn(2,4,6)); self.assertTrue((features>0).all()); self.assertEqual(model.random_projection.shape,(6,6))
    def test_gagnn_group_assignments_are_probabilities(self):
        model=factories()["GAGNN"](); model(torch.randn(2,6,4),marks(2,6)); assignment=model.layers[0].last_assignment; self.assertIsNotNone(assignment); torch.testing.assert_close(assignment.sum(-1),torch.ones(2,4))
    def test_mtgnn_directed_constructor_and_mixhop(self):
        model=factories()["MTGNN"](); adjacency=model.graph_constructor(); self.assertEqual(adjacency.shape,(4,4)); self.assertFalse(torch.allclose(adjacency,adjacency.T)); self.assertEqual(model.layers[0].forward_graph.depth,1)
    def test_megacrn_memory_queries_form_meta_graph(self):
        model=factories()["MegaCRN"](); model(torch.randn(2,6,4)); torch.testing.assert_close(model.last_memory_attention.sum(-1),torch.ones(2,4)); torch.testing.assert_close(model.last_meta_graph.sum(-1),torch.ones(2,4))
    def test_pm25_graph_messages_drive_both_gru_gates(self):
        model=factories()["PM25_GNN"](); self.assertEqual(model.encoder.gates.in_features,2*(8+8)); self.assertEqual(model.decoder.gates.out_features,16)
    def test_stgode_has_dual_graph_ode_branches(self):
        model=factories()["STGODE"](); self.assertFalse(torch.equal(model.spatial_graph,model.semantic_graph)); self.assertEqual(model.spatial_branch.ode.steps,1); self.assertEqual(model.semantic_branch.ode.steps,1)

class CompleteRuntimeTests(unittest.TestCase):
    def test_forward_backward_finite_all_active_and_round_trip(self):
        torch.manual_seed(260827)
        for name,factory in factories().items():
            with self.subTest(model=name):
                model=factory().cpu().eval(); x=torch.randn(2,6,4,requires_grad=True); output=model(x,marks(2,6)); self.assertEqual(output.shape,(2,3,4)); self.assertTrue(torch.isfinite(output).all()); output.square().mean().backward(); self.assertGreater(x.grad.abs().max().item(),0)
                for parameter_name,parameter in model.named_parameters(): self.assertIsNotNone(parameter.grad,parameter_name); self.assertTrue(torch.isfinite(parameter.grad).all(),parameter_name); self.assertGreater(parameter.grad.abs().max().item(),0,parameter_name)
                clone=factory().eval(); clone.load_state_dict(copy.deepcopy(model.state_dict()),strict=True); torch.testing.assert_close(clone(x.detach(),marks(2,6)),model(x.detach(),marks(2,6)))
    def test_batch_sequence_node_and_adjacency_boundaries(self):
        single=np.ones((1,1),dtype=np.float32)
        boundary=factories(1,1,1,single)
        for name,factory in boundary.items():
            with self.subTest(model=name): self.assertEqual(factory().eval()(torch.randn(1,1,1)).shape,(1,1,1))
        for name,factory in factories().items():
            with self.subTest(model=name):
                model=factory(); self.assertEqual(model(torch.randn(1,6,4),marks(1,6)).shape,(1,3,4))
                with self.assertRaises(ValueError): model(torch.randn(1,5,4))
        for name,factory in factories(adj=np.eye(3,dtype=np.float32)).items():
            with self.subTest(model=name):
                with self.assertRaises(ValueError): factory()
    def test_marks_and_adjacency_contracts_are_active(self):
        x=torch.randn(2,6,4)
        for name,factory in factories().items():
            with self.subTest(model=name):
                model=factory().eval(); self.assertGreater((model(x,marks(2,6,7))-model(x)).abs().max().item(),0)
        for name in factories():
            with self.subTest(model=name):
                torch.manual_seed(91); identity=factories(adj=np.eye(4,dtype=np.float32))[name]().eval()
                torch.manual_seed(91); connected=factories(adj=graph())[name]().eval()
                self.assertGreater((identity(x)-connected(x)).abs().max().item(),0)

if __name__=="__main__": unittest.main()
