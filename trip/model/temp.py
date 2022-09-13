def forward(self, features: Tensor, invariant_edge_feats: Tensor, basis: Tensor):
    # features: nil
    # radial profile: noif
    # basis: nlfk

    with nvtx_range(f'VersatileConvSE3'):
        num_edges = features.shape[0]
        in_dim = features.shape[2]
        with nvtx_range(f'RadialProfile'):
            radial_weights = self.radial_func(invariant_edge_feats) \
                .view(-1, self.channels_out, self.channels_in * self.freq_sum) # rad: no(if)

        if basis is not None:
            # This block performs the einsum n i l, n o i f, n l f k -> n o k
            basis_view = basis.view(num_edges, in_dim, -1) # basis: nl(fk)
            tmp = (features @ basis_view).view(num_edges, -1, basis.shape[-1]) # tmp: ni(fk) -> n(if)k # This is point where things go wrong!
            return radial_weights @ tmp 
        else:
            # k = l = 0 non-fused case
            return radial_weights @ features
