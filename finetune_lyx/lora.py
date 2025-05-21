import torch
import transformers

from utils import recursive_getattr, recursive_setattr


class LoRALinear(torch.nn.Module):
    def __init__(self, weight, bias, lora_dim, lora_scaling):
        super(LoRALinear, self).__init__()
        # Save original weight and bias
        self.weight = torch.nn.Parameter(weight)        # (out_dim, in_dim)
        self.bias = torch.nn.Parameter(bias)
        # TODO: Implement lora left and right weights
        self.lora_right_weight = torch.nn.Parameter(torch.randn(lora_dim, self.weight.shape[-1]))   # A matrix
        self.lora_left_weight = torch.nn.Parameter(torch.zeros((self.weight.shape[0], lora_dim)))   # B matrix
        #############################################
        self.lora_scaling = lora_scaling / lora_dim
        self.init_parameters()
        # TODO: Freeze original weight and bias
        #
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        #######################################

    def init_parameters(self):
        # TODO: Initialize LoRA parameters
        # raise NotImplementedError
        torch.nn.init.zeros_(self.lora_left_weight)
        torch.nn.init.normal_(self.lora_right_weight)
        ##################################

    def forward(self, input):
        # TODO: Implement the forward function
        # raise NotImplementedError
        size_out = input.size()[:-1] + (self.weight.shape[0],)
        weight_after = torch.addmm(self.weight, self.lora_left_weight , self.lora_right_weight)
        output = torch.addmm(self.bias, input.view(-1, input.size(-1)), weight_after.t())
        output = output.view(size_out)
        return output
        ######################################


def convert_linear_layer_to_lora(model: transformers.GPT2LMHeadModel, part_module_name, lora_dim=0, lora_scaling=1):
    replace_name = []
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Linear) or isinstance(module, transformers.pytorch_utils.Conv1D)) and part_module_name in name:
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        if isinstance(module, torch.nn.Linear):
            tmp = LoRALinear(module.weight, module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        elif isinstance(module, transformers.pytorch_utils.Conv1D):
            tmp = LoRALinear(module.weight.t().detach(), module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        else:
            raise ValueError("Unsupported module type")
        recursive_setattr(model, name, tmp)
    return model


def only_optimize_lora_parameters(model: transformers.GPT2LMHeadModel):
    # TODO: Turn off the gradient of all the parameters except the LoRA parameters
    # raise NotImplementedError
    for n, m in model.named_modules():
        if not isinstance(m, LoRALinear):
            m.requires_grad_(False)
    return model
    ##############################################################################

def get_lora_state_dict(model: transformers.GPT2LMHeadModel):
    # TODO: return lora left and right weights as state dict
    # The saved state dict will be used later for loading
    # raise NotImplementedError
    sd = {}
    for n, m in model.named_modules():
        if isinstance(m, LoRALinear):
            m_sd = m.state_dict()
            k_list = m_sd.keys()
            for k in k_list:
                sd['.'.join([n, k])] = m_sd
    return sd            
    ########################################################