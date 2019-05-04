class DenseStackedConv1dMaskRegression(nn.Module):
    def __init__(self,  
                 pretrained_encoder,
                 pretrained_decoder,
                 depth=1,
                 n_basis=None,
                 embed_factor=2,
                 dropout_rate=0.0, 
                 groups='isolated',
                 bottleneck_groups='isolated',
                 kernel_size=1,
                 n_sources=2):
        super().__init__()
        self.n_sources = 2 
        self.depth = depth 
        self.embed_factor = embed_factor
        self.dropout_rate = dropout_rate
        
        self.n_basis = n_basis
        
        if groups == 'full':
            self.groups = 1 
        elif groups == 'isolated':
            self.groups = embed_factor * self.n_basis
            
        if bottleneck_groups == 'full':
            self.bottleneck_groups = 1 
        elif bottleneck_groups == 'isolated':
            self.bottleneck_groups = self.n_basis        
            
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size-1)//2 
        
        
        self.encoder = pretrained_encoder
        self.decoder = pretrained_decoder
        
        self.encoder.conv.weight.requires_grad = False 
        self.encoder.conv.bias.requires_grad = False
        self.decoder.deconv.weight.requires_grad = False 
        self.decoder.deconv.bias.requires_grad = False 
        
#         self.encoder.conv.weight.requires_grad = True 
#         self.encoder.conv.bias.requires_grad = True
#         self.decoder.deconv.weight.requires_grad = True 
#         self.decoder.deconv.bias.requires_grad = True 

#       now create the actual net with blocks
        self.bottleneck_conv = nn.Conv1d(self.n_basis,
                                         embed_factor * self.n_basis,
                                         self.kernel_size,
                                         padding=self.padding,
                                         dilation=1,
                                         groups=self.bottleneck_groups)
    
        self.mask_est_modules = nn.ModuleList()
        for i in range(self.depth):
            self.mask_est_modules.append(nn.Softplus())
            self.mask_est_modules.append(nn.Dropout(self.dropout_rate))

            self.mask_est_modules.append(nn.BatchNorm1d(self.embed_factor * self.n_basis))
            self.mask_est_modules.append(nn.Conv1d(
                                         self.embed_factor * self.n_basis,
                                         self.embed_factor * self.n_basis,
                                         self.kernel_size,
                                         padding=self.padding,
                                         dilation=1,
                                         groups=self.groups))
            
        self.out_bottleneck_conv = nn.Conv1d(embed_factor * self.n_basis,
                                             self.n_sources * self.n_basis,
                                             self.kernel_size,
                                             padding=self.padding,
                                             dilation=1,
                                             groups=self.bottleneck_groups)

    def get_encoded_mixture(self, mixture_wav):
        return self.encoder(mixture_wav)
    
    def get_estimated_masks(self, enc_mixture):
        x = self.bottleneck_conv(enc_mixture)
        for i in range(self.depth):
            x = self.mask_est_modules[i](x)
        x = self.out_bottleneck_conv(x)
        
#         x = x + 0.1 + 0.005 * torch.randn_like(x)
        x = F.softplus(x)
        
        sources_masks = x.unsqueeze(1).contiguous().view(x.shape[0], 
                                                         self.n_sources, 
                                                         -1, 
                                                         x.shape[-1])
        return F.softmax(sources_masks, dim=1)
#         sources_masks = sources_masks / (torch.sum(sources_masks, dim=1, keepdim=True) + 10e-8)
#         sources_masks[:, 0, :, :] = 1. - sources_masks[:, 1, :, :] 
        
#         return sources_masks 
        
    
    def forward(self, mixture_wav):
        enc_mixture = self.get_encoded_mixture(mixture_wav)
        sources_masks = self.get_estimated_masks(enc_mixture)
        return sources_masks
    
    def infer_source_signals(self, 
                             mixture_wav,
                             sources_masks=None):
        enc_mixture = self.get_encoded_mixture(mixture_wav)
        if sources_masks is None:
            m_out = self.get_estimated_masks(enc_mixture)
        else:
            m_out = sources_masks
            
        return torch.cat([self.decoder(enc_mixture * m_out[:, c, :, :]) 
                          for c in range(self.n_sources)], dim=1)
    
    def AE_recontruction(self, mixture_wav):
        enc_mixture = self.encoder(mixture_wav)
        return self.decoder(enc_mixture)
    
    
# simple_dence_conv1d = DenseStackedConv1dMaskRegression(stft_encoder,
#                                                        stft_decoder,
#                                                        n_basis=n_fft,
#                                                        n_sources=2,
#                                                        depth=5)
# numparams = 0
# for f in simple_dence_conv1d.parameters():
#     if f.requires_grad == True:
#         numparams += f.numel()
# print(numparams)
# print(simple_dence_conv1d)
