"""Self-contained representation-shape losses and weak adapter."""
import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
GAMMAS=(0.125,0.25,0.5); KAPPAS=(1.0,2.0,4.0)
class WeakAdapter(nn.Module):
 def __init__(self,input_dim=256,hidden_dim=128,heads=4):
  super().__init__(); self.norm=nn.LayerNorm(input_dim,elementwise_affine=False); self.query=nn.Parameter(torch.randn(1,1,hidden_dim)*.02); self.attention=nn.MultiheadAttention(hidden_dim,heads,kdim=input_dim,vdim=input_dim,batch_first=True,dropout=0.0); self.output=nn.Linear(hidden_dim,3)
 def forward(self,tokens):
  q=self.query.expand(len(tokens),-1,-1); pooled,_=self.attention(q,self.norm(tokens),self.norm(tokens),need_weights=False); return self.output(pooled[:,0])
@dataclass(frozen=True)
class Shape:
 name:str; dim:int=3; nu:float=5.0
 @property
 def laplace_scale(self): return 1/math.sqrt(self.dim+1)
 @property
 def student_scale(self): return math.sqrt((self.nu-2)/self.nu)
 @property
 def sphere_radius(self): return math.sqrt(self.dim)
def pairwise_squared(a,b): return (a.square().sum(-1,keepdim=True)+b.square().sum(-1).unsqueeze(0)-2*a@b.T).clamp_min(0)
def gaussian_mmd(z):
 dist=pairwise_squared(z,z); norm2=z.square().sum(-1); dim=z.shape[-1]; vals=[]
 for g in GAMMAS:
  emp=torch.exp(-g*dist).mean(); cross=(1+2*g)**(-dim/2)*torch.exp(-g*norm2/(1+2*g)).mean(); target=(1+4*g)**(-dim/2); vals.append(emp-2*cross+target)
 return torch.stack(vals).mean()
def _score(z,s):
 r2=z.square().sum(-1,keepdim=True)
 if s.name=='radial_laplace': return -z/(s.laplace_scale*(r2+1e-6).sqrt())
 if s.name=='student_t': return -(s.nu+s.dim)*z/(s.nu*s.student_scale**2+r2)
 raise ValueError(s.name)
def gaussian_kernel_ksd(z,s):
 delta=z[:,None]-z[None]; r2=delta.square().sum(-1); sc=_score(z,s); dot=sc@sc.T; sd=(delta*(sc[:,None]-sc[None])).sum(-1); vals=[]
 for g in GAMMAS:
  k=torch.exp(-g*r2); vals.append((k*(dot+2*g*sd+2*g*s.dim-4*g*g*r2)).mean())
 return torch.stack(vals).mean()
def sphere_loss(z,s):
 u=F.normalize(z,dim=-1); sim=u@u.T; off=~torch.eye(len(z),dtype=torch.bool,device=z.device); uni=torch.stack([torch.exp(k*(sim[off]-1)).mean() for k in KAPPAS]).mean(); return uni+(z.norm(dim=-1)-s.sphere_radius).square().mean()
def target_discrepancy(z,s):
 if s.name=='gaussian': return gaussian_mmd(z)
 if s.name in ('radial_laplace','student_t'): return gaussian_kernel_ksd(z,s)
 if s.name=='sphere': return sphere_loss(z,s)
 raise ValueError(s.name)
def distance_preservation(z,source):
 a=torch.pdist(z); b=torch.pdist(source); return (a/a.mean().clamp_min(1e-6)-b/b.mean().clamp_min(1e-6)).square().mean()
def full_loss(z,source,s,initial_discrepancy):
 return target_discrepancy(z,s)/max(abs(initial_discrepancy),1e-4)+.05*distance_preservation(z,source)
