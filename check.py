import torch
print(torch.cuda.get_device_name())  # GPU 모델 확인
print(torch.backends.cuda.flash_sdp_enabled())  # Flash Attention 지원 여부
