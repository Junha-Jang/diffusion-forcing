# import config
import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from .cldm.model import create_model, load_state_dict
from .cldm.ddim_hacked import DDIMSampler

import torch.nn as nn

class StableDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model_config_path='./algorithms/controlnet/models/cldm_v15.yaml'
        self.model_config_path='./algorithms/controlnet/models/cldm_v21.yaml'
        # self.model_weights_path='./algorithms/controlnet/models/control_sd15_canny.pth'
        self.model_weights_path='./algorithms/controlnet/models/control_sdn21_ini.ckpt'

        # Configs (for training)
        self.learning_rate = 1e-5
        self.sd_locked = True
        self.only_mid_control = False

        self._build_model()
    
    def _build_model(self):
        self.model = create_model(self.model_config_path).cpu()
        self.model.load_state_dict(load_state_dict(self.model_weights_path, location='cuda'))

        self.model.learning_rate = self.learning_rate
        self.model.sd_locked = self.sd_locked
        self.model.only_mid_control = self.only_mid_control

        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)

    def process(
        self, 
        input_image, 
        control, 
        prompt='minecraft screenshot',
        a_prompt='best quality, extremely detailed',
        n_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
        num_samples=1,
        image_resolution=128,
        ddim_steps=2,
        guess_mode=False,
        strength=1.0,
        scale=9.0,
        seed=-1,
        eta=0.0,
    ):
        with torch.no_grad():
            img = input_image
            # print(f'Image shape: {img.shape}')
            B, C, H, W = img.shape

            # print(f'Image shape: {img.shape}')

            c_concat = [control]
            c_crossattn = [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
            # print(f'c_concat shape: {c_concat[0].shape}')
            # print(f'c_crossattn shape: {c_crossattn[0].shape}')

            cond = {
                "c_concat": c_concat,
                "c_crossattn": c_crossattn
            }
            un_cond = {
                "c_concat": None if guess_mode else c_concat,
                "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]
            }
            shape = (4, H // 8, W // 8)

            # if config.save_memory:
            #     self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                             shape, cond, verbose=False, x_T=input_image, eta=eta,
                                                             unconditional_guidance_scale=scale,
                                                             unconditional_conditioning=un_cond)

            # if config.save_memory:
            #     self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            # x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            for i, sample in enumerate(samples):
                cv2.imwrite(f'output_image_{i}.png', sample)

            # results = [x_samples[i] for i in range(num_samples)]
        # return results
        return x_samples

if __name__ == '__main__':
    # 예시 사용
    input_image = np.random.randint(0, 255, (512, 512, 3)).astype(np.uint8)
    control = torch.rand(1, 3, 512, 512).cuda()

    # 클래스 초기화
    model_instance = StableDiffusionModel()
    results = model_instance.process(input_image, control)
