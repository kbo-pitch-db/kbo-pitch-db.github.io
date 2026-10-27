import importlib.util,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pitch_optimizer import optimize


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch environment required')
class PitchOptimizerTests(unittest.TestCase):
    def run_fit(self,device,graph):
        import torch
        p=torch.nn.Parameter(torch.tensor([1.,-.8],device=device))
        optimizer=torch.optim.Adam([p],lr=.04,capturable=device.startswith('cuda'))
        def evaluate():
            loss=2.+((p-.15)**2).sum()
            return loss,{},()
        loss,best=optimize([p],optimizer,evaluate,40,device,graph)
        self.assertGreaterEqual(loss,2.,'a captured but unexecuted loss cannot be selected')
        self.assertEqual(float(optimizer.state[p]['step']),40.,'warmup and graph recapture do not change iteration count')
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'],.012)
        self.assertAlmostEqual(float(2.+((best[0]-.15)**2).sum()),loss,places=5,msg='best parameters describe the saved pre-update loss')
        return loss,best[0].cpu()

    def test_eager_tracks_the_best_matching_parameters(self):
        self.run_fit('cpu',False)

    def test_cuda_graph_matches_eager_through_learning_rate_change(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        device='cuda:1' if torch.cuda.device_count()>1 else 'cuda:0'
        a,x=self.run_fit(device,False);b,y=self.run_fit(device,True)
        self.assertAlmostEqual(a,b,places=6);torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()
