import torch


class PerpCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model):
        def custom_cfg_function(args):
            cond = args["cond_denoised"]
            uncond = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            x = args["input"]

            cond_norm = torch.nn.functional.normalize(cond, dim=[-1, -2, -3])
            uncond_parallel = (uncond * cond_norm).sum(dim=[-1, -2, -3], keepdim=True) * cond_norm
            uncond_orthogonal = uncond - uncond_parallel

            return (x - (cond - (cond_scale * uncond_orthogonal)))
        
        m = model.clone()
        m.set_model_sampler_cfg_function(custom_cfg_function)

        return (m, )

NODE_CLASS_MAPPINGS = {
    "PerpCFG": PerpCFG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerpCFG": "PerpCFG",
}
