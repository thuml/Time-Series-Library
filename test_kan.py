from models.KANFormer import KANFormer

# Example configuration
config = {
    'vocabulary_size': 10000,
    'hidden_size': 512,
    'num_layers': 6,
    'num_heads': 8,
    'dropout': 0.1,
    'max_length': 512,
    'num_experts': 10,
    'n_experts_per_token': 2,
    'd_ff': 2048
}
model = KANFormer(config)
print(model)
