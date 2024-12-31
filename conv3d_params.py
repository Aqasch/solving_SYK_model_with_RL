

def calculate_output_dimensions(input_dims, conv_params, pool_params):
    # Unpack input dimensions
    D_in, H_in, W_in = input_dims
    
    # Unpack convolution parameters
    num_channels, kernel_size, stride, padding = conv_params
    K_d, K_h, K_w = kernel_size
    S_d, S_h, S_w = stride
    P_d, P_h, P_w = padding
    
    # Calculate output dimensions after convolution
    D_out = (D_in + 2 * P_d - K_d) // S_d + 1
    H_out = (H_in + 2 * P_h - K_h) // S_h + 1
    W_out = (W_in + 2 * P_w - K_w) // S_w + 1
    
    # Unpack pooling parameters
    pool_kernel_size, pool_stride = pool_params
    PK_d, PK_h, PK_w = pool_kernel_size
    PS_d, PS_h, PS_w = pool_stride
    
    # Calculate output dimensions after pooling
    D_out = (D_out - PK_d) // PS_d + 1
    H_out = (H_out - PK_h) // PS_h + 1
    W_out = (W_out - PK_w) // PS_w + 1
    
    return D_out, H_out, W_out

def calculate_total_features(input_dims, conv_layers, pool_layers):
    current_dims = input_dims
    for i in range(len(conv_layers)):
        conv_params = conv_layers[i]
        pool_params = pool_layers[i]
        
        # Calculate output dimensions after conv and pool layers
        current_dims = calculate_output_dimensions(current_dims, conv_params, pool_params)
    
    # The last convolution layer's number of channels
    final_num_channels = conv_layers[-1][0]
    
    # Calculate total features
    total_features = final_num_channels * current_dims[0] * current_dims[1] * current_dims[2]
    return total_features

# Input dimensions (D, H, W)
input_dims = (1, 1, 1)  # Example input dimensions

# Parameters for each Conv3d layer: (num_channels, kernel_size, stride, padding)
conv_layers = [
    (32, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
    (64, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
    (128, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
]

# Parameters for each MaxPool3d layer: (kernel_size, stride)
pool_layers = [
    ((1, 1, 1), (2, 2, 2)),
    ((1, 1, 1), (2, 2, 2)),
    ((1, 1, 1), (2, 2, 2)),
]

# Calculate total number of features
total_features = calculate_total_features(input_dims, conv_layers, pool_layers)
print(f'Total number of features: {total_features}')
