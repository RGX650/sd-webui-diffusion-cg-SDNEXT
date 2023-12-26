from dataclasses import dataclas
from modules.sd_samplers_kdiffusion import KDiffusionSampler
from scripts.cg_version import VERSION
from modules import script_callbacks
import modules.scripts as scripts
from modules import shared
import gradio as gr


DYNAMIC_RANGE = [3.25, 2.5, 2.5, 2.5]

Default_LUTs = {
    'C': 0.01,
    'M': 0.5,
    'Y': -0.13,
    'K': 0
}


def normalize_tensor(x, r):
    X = x.detach().clone()

    ratio = r / max(abs(float(X.min())), abs(float(X.max())))
    X *= max(ratio, 0.99)

    return X


original_callback = KDiffusionSampler.callback_state

def center_callback(self, d):
    options: "DiffusionCG.CGOptions" = self.diffcg_options

    if not options.is_enabled():
        return original_callback(self, d)

    batchSize = d['x'].size(0)
    channels = len(self.LUTs)

    X = d['x'].detach().clone()
    Y = d[self.diffcg_tensor].detach().clone()

    for b in range(batchSize):
        for c in range(channels):

            if self.diffcg_recenter_strength > 0.0:
                d['x'][b][c] += (self.LUTs[c] - X[b][c].mean()) * self.diffcg_recenter_strength

            if self.diffcg_normalize and (d['i'] + 1) >= self.diffcg_last_step - 1:
                d[self.diffcg_tensor][b][c] = normalize_tensor(Y[b][c], DYNAMIC_RANGE[c])

    return original_callback(self, d)

KDiffusionSampler.callback_state = center_callback


# ["None", "txt2img", "img2img", "Both"]
ac = getattr(shared.opts, 'always_center', 'None')
an = getattr(shared.opts, 'always_normalize', 'None')
def_sd = getattr(shared.opts, 'default_arch', '1.5')

c_t2i = (ac == "txt2img" or ac == "Both")
c_i2i = (ac == "img2img" or ac == "Both")
n_t2i = (an == "txt2img" or an == "Both")
n_i2i = (an == "img2img" or an == "Both")


class DiffusionCG(scripts.Script):
    
    @dataclass
    class CGOptions:
        enable_centering: bool
        enable_normalization: bool

        def is_enabled(self):
            return self.enable_normalization or self.enable_centering

    options = CGOptions

    
    def title(self):
        return "DiffusionCG"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(f'Diffusion CG {VERSION}', open=False):
            
            with gr.Row():
                enableG = gr.Checkbox(label="Enable (Global)")
                sd_ver = gr.Radio(['1.5', 'XL'], value='1.5', label="Stable Diffusion Version")
                
            with gr.Row():
                with gr.Group():
                    gr.Markdown('<h3 align="center">Recenter</h3>')
                    enableC = gr.Checkbox(label="Enable")
                    
                    if not is_img2img:
                        v = 1.0 if c_t2i else 0.0
                    else:
                        v = 1.0 if c_i2i else 0.0

                    rc_str = gr.Slider(label="Effect Strength", minimum=0.0, maximum=1.0, step=0.2, value=v)
                    
                with gr.Group():
                    gr.Markdown('<h3 align="center">Normalization</h3>')
                    enableN = gr.Checkbox(label="Enable")

            with gr.Accordion('Recenter Settings', open=False):
                with gr.Group(visible=(def_sd=='1.5')) as setting15:
                    C = gr.Slider(label="C", minimum=-1.00, maximum=1.00, step=0.01, value=Default_LUTs['C'])
                    M = gr.Slider(label="M", minimum=-1.00, maximum=1.00, step=0.01, value=Default_LUTs['M'])
                    Y = gr.Slider(label="Y", minimum=-1.00, maximum=1.00, step=0.01, value=Default_LUTs['Y'])
                    K = gr.Slider(label="K", minimum=-1.00, maximum=1.00, step=0.01, value=Default_LUTs['K'])

                with gr.Group(visible=(def_sd=='XL')) as settingXL:
                    L = gr.Slider(label="L", minimum=-1.00, maximum=1.00, step=0.01, value=0.0)
                    a = gr.Slider(label="a", minimum=-1.00, maximum=1.00, step=0.01, value=0.0)
                    b = gr.Slider(label="b", minimum=-1.00, maximum=1.00, step=0.01, value=0.0)

            def on_radio_change(choice):
                if choice != "1.5":
                    return [gr.Group.update(visible=True), gr.Group.update(visible=False)]
                else:
                    return [gr.Group.update(visible=False), gr.Group.update(visible=True)]

            sd_ver.select(on_radio_change, sd_ver, [setting15, settingXL])

        return [enableG, sd_ver, rc_str, enableC, enableN, C, M, Y, K, L, a, b]


    def before_hr(self, p, *args):
        KDiffusionSampler.diffcg_options.enable_normalzation = False
        
    
    def process(self, p, enableG:bool, sd_ver:str, rc_str:float, enableC:bool, enableN:bool, C, M, Y, K, L, a, b):
        self.options = DiffusionCG.CGOptions(enable_centering = enableC, enable_normalization = enableN)
        
        KDiffusionSampler.diffcg_options = self.options
        KDiffusionSampler.diffcg_enable = enableG
        KDiffusionSampler.diffcg_recenter = enableC
        KDiffusionSampler.diffcg_normalize = enableN

        if sd_ver == '1.5':
            KDiffusionSampler.LUTs = [-K, -M, C, Y]
        else:
            KDiffusionSampler.LUTs = [L, -a, b]
        
        KDiffusionSampler.diffcg_tensor = 'x' if p.sampler_name.strip() == 'Euler' else 'denoised'
        KDiffusionSampler.diffcg_recenter_strength = rc_str if enableC else 0

        if not hasattr(p, 'enable_hr') and hasattr(p, 'denoising_strength') and not shared.opts.img2img_fix_steps and p.denoising_strength < 1.0:
            KDiffusionSampler.diffcg_last_step = int(p.steps * p.denoising_strength) + 1
        else:
            KDiffusionSampler.diffcg_last_step = p.steps


def restore_callback():
    KDiffusionSampler.callback_state = original_callback

script_callbacks.on_script_unloaded(restore_callback)
