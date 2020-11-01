import pytest
import gputorch

@pytest.mark.needs_gpu
def test_cuda_check():
    assert gputorch.cuda_check() == True

def test_train():
    total_epochs = 2
    PATH = "./model.pth"
    assert gputorch.train(total_epochs, PATH) < 1.3

def test_test():
    PATH = "./model.pth"
    assert gputorch.test(PATH) > 40
