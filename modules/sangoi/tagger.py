from contextlib import contextmanager
import torch, tkinter as tk
from tkinter import filedialog
from PIL import Image
from pathlib import Path
import open_clip
from clip_interrogator import Config, Interrogator

# ====== CONFIGURAÇÃO GERAL ======
CHECKPOINT = Path(r"F:\stable-diffusion-webui-forge\models\diffusers\models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k\snapshots\743c27bd53dfe508a0ade0f50698f99b39d03bec\open_clip_pytorch_model.bin")
MODEL_NAME = "ViT-bigG-14"                       # arquitetura
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAPTION_MODEL = 'blip-large'                     # ou 'blip-base' p/ ~1 GB

MODES = ['caption', 'best', 'fast', 'classic', 'negative']

@contextmanager
def no_grad():
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        with torch.inference_mode():
            yield
    finally:
        torch.set_grad_enabled(prev)

# -------- utilidades GUI ---------
def pick_image() -> Path:
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    fn = filedialog.askopenfilename(
        title="Selecione uma imagem",
        filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.webp;*.bmp")]
    )
    root.destroy()
    return Path(fn) if fn else None
# ---------------------------------

# -------- CLIP custom ------------
def load_custom_clip():
    model, preprocess, tokenizer = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=False, device=DEVICE
    )
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model = model.to(device=DEVICE, dtype=torch.float16).eval()
    return model, preprocess, tokenizer
# ---------------------------------

# -------- CI style “load/unload” --
_ci = None
def load_ci():
    global _ci
    if _ci is not None:
        return _ci

    clip_model, clip_preprocess, tokenizer = load_custom_clip()

    cfg = Config()
    cfg.device = DEVICE                       # CLIP na GPU
    cfg.clip_model = clip_model
    cfg.clip_preprocess = clip_preprocess
    cfg.clip_model_name = f"{MODEL_NAME}/laion2b_s39b_b160k"
    cfg.caption_model_name = CAPTION_MODEL
    cfg.quiet = True
    _ci = Interrogator(cfg)

    # --- Monkey-patch: embeddings de texto sempre no CPU -------------
    import types, torch.nn.functional as F
    def _encode_text_cpu(self, texts):
        self._load_clip_model()
        txt = self.tokenizer(texts).to('cpu')
        with torch.inference_mode():
            return self.clip_model.encode_text(txt.to(self.clip_model.text_projection.dtype)).to('cpu')
    _ci._encode_text = types.MethodType(_encode_text_cpu, _ci)
    # ----------------------------------------------------------------
    return _ci
# ---------------------------------

def unload_ci():
    global _ci
    if _ci is None:
        return
    _ci.caption_model = _ci.caption_model.to('cpu')
    _ci.clip_model    = _ci.clip_model.to('cpu')
    _ci.caption_offloaded = True
    _ci.clip_offloaded    = True
    torch.cuda.empty_cache()
    _ci = None
# ---------------------------------

def interrogate(image: Image.Image, mode: str) -> str:
    ci = load_ci()
    image = image.convert('RGB')
    with no_grad():                      # <<< linha nova
        if mode == 'best':
            return ci.interrogate(image)
        if mode == 'caption':
            return ci.generate_caption(image)
        if mode == 'classic':
            return ci.interrogate_classic(image)
        if mode == 'fast':
            return ci.interrogate_fast(image)
        if mode == 'negative':
            return ci.interrogate_negative(image)
    raise ValueError(f"Modo desconhecido: {mode}")

# ======== MAIN ========
if __name__ == "__main__":
    img_path = pick_image()
    if not img_path:
        print("Nenhuma imagem escolhida – saindo.")
        exit()

    # Escolher modo pelo terminal
    print(f"Modos disponíveis: {', '.join(MODES)}")
    mode = input("Escolha o modo (enter = best): ").strip().lower() or 'best'
    if mode not in MODES:
        print(f"Modo '{mode}' inválido."); exit()

    try:
        img = Image.open(img_path)
        print("\nGerando… aguarde.")
        prompt = interrogate(img, mode)
        print("\n— Prompt —")
        print(prompt)
    except torch.cuda.OutOfMemoryError:
        print("🔥 Sem VRAM suficiente.")
    finally:
        unload_ci()
