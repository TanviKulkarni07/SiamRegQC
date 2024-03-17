import torch

def register_hooks(model):
    activations = []

    def hook_fn(module, input, output):
        activations.append(output)

    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    return activations, hooks

# Function to remove hooks
def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()

def get_activations(model, input1, input2, device):
    
    input1 = input1.type(torch.FloatTensor).to(device)
    input2 = input2.type(torch.FloatTensor).to(device)
    
    layer_num = 7
    activations, hooks = register_hooks(model.shared_layer1)
    with torch.no_grad():
        model.shared_layer1(input1)

    remove_hooks(hooks)
    feature_maps1_dis = activations[layer_num][0].cpu().numpy()

    activations, hooks = register_hooks(model.shared_layer1)
    with torch.no_grad():
        model.shared_layer1(input2)

    remove_hooks(hooks)
    feature_maps2_dis = activations[layer_num][0].cpu().numpy()
    
    return feature_maps1_dis, feature_maps2_dis

def get_activations_encoder(model, input1, device):
    
    input1 = input1.type(torch.FloatTensor).to(device)
    
    layer_num = 7
    activations, hooks = register_hooks(model)
    with torch.no_grad():
        model(input1)

    remove_hooks(hooks)
    feature_maps1_dis = activations[layer_num][0].cpu().numpy()
    
    return feature_maps1_dis 