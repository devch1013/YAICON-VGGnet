import torch.nn as nn
import torch.nn.functional as F


class BindNetwork(nn.Module):
    def __init__(self, **cfgs):
        super(BindNetwork, self).__init__()
        # W0 transformation
        self.image_bind_proj =  nn.Linear(1024, 2048)
        bridge_norm_layer = nn.LayerNorm
        bridge_bias = True
        self.image_bind_norm_1 = bridge_norm_layer(2048)
        self.image_bind_f1_1 = nn.Linear(2048, 2048 * 4, bias=bridge_bias)
        self.image_bind_f2_1 = nn.Linear(2048 * 4, 2048, bias=bridge_bias)
        self.image_bind_f3_1 = nn.Linear(2048, 2048 * 4, bias=bridge_bias)

        self.image_bind_norm_2 = bridge_norm_layer(2048)
        self.image_bind_f1_2 = nn.Linear(2048, 2048 * 4, bias=bridge_bias)
        self.image_bind_f2_2 = nn.Linear(2048 * 4, 2048, bias=bridge_bias)
        self.image_bind_f3_2 = nn.Linear(2048, 2048 * 4, bias=bridge_bias)

        self.image_bind_norm_3 = bridge_norm_layer(2048)
        self.image_bind_f1_3 = nn.Linear(2048, 2048 * 4, bias=bridge_bias)
        self.image_bind_f2_3 = nn.Linear(2048 * 4, 2048, bias=bridge_bias)
        self.image_bind_f3_3 = nn.Linear(2048, 2048 * 4, bias=bridge_bias)
        
        self.image_bind_norm_final = bridge_norm_layer(2048)
        
        self.final = nn.Linear(2048, 77*4096, bias=bridge_bias)
        
        self.back_1 = nn.Linear(77*4096, 2048)
        self.back_2 = nn.Linear(2048, 1024)
        self.back_3 = nn.Linear(1024, 1024)
        self.back_4 = nn.Linear(1024, 1024)
        
        
    def backward_net(self, output):
        x = F.relu(self.back_1(output))
        x = F.relu(self.back_2(x))
        x = F.relu(self.back_3(x))
        x = self.back_4(x)
        
        return x
        
        
    def forward(self, x):
        
        visual_feats = self.image_bind_proj(x)
        visual_feats_norm = self.image_bind_norm_1(visual_feats)
        visual_feats = visual_feats + self.image_bind_f2_1(F.silu(self.image_bind_f1_1(visual_feats_norm)) * self.image_bind_f3_1(visual_feats_norm))

        visual_feats_norm = self.image_bind_norm_2(visual_feats)
        visual_feats = visual_feats + self.image_bind_f2_2(F.silu(self.image_bind_f1_2(visual_feats_norm)) * self.image_bind_f3_2(visual_feats_norm))

        visual_feats_norm = self.image_bind_norm_3(visual_feats)
        visual_feats = visual_feats + self.image_bind_f2_3(F.silu(self.image_bind_f1_3(visual_feats_norm)) * self.image_bind_f3_3(visual_feats_norm))
        visual_feats = self.image_bind_norm_final(visual_feats)
        visual_feats = self.final(visual_feats)
        
        x = visual_feats.view(-1, 77, 4096)
        
        reverse_result = self.backward_net(visual_feats)

        return x, reverse_result