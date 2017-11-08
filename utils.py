from torch.autograd import Variable
import torch
def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    # if torch.cuda.is_available():
        # tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)