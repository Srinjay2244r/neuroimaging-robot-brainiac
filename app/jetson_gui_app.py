# BrainIAC Jetson demo GUI (Gradio). Runs the TensorRT engine on the Jetson and shows prediction, probabilities, slices and latency.
#   python3 app.py --engine brainiac_fp16.engine --spec spec.json --samples samples/ --labels labels.csv --host 0.0.0.0 --port 7860
# No TensorRT yet? use --onnx brainiac_multiclass.onnx (ONNX Runtime) to test the GUI on any machine.
import os, sys, json, time, argparse, glob
import numpy as np, torch, torch.nn.functional as F, nibabel as nib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import gradio as gr

ap = argparse.ArgumentParser()
ap.add_argument("--engine"); ap.add_argument("--onnx"); ap.add_argument("--spec", default="spec.json")
ap.add_argument("--samples", help="folder with preprocessed demo scans (.nii/.nii.gz)"); ap.add_argument("--labels", help="optional CSV with columns path,label (matched by file name) to show the true label")
ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=7860); ap.add_argument("--share", action="store_true")
args = ap.parse_args()
SPEC = json.load(open(args.spec)); CLS = SPEC["classes"]; NICE = {"normal": "Normal", "mci": "MCI", "alzheimers": "Alzheimer's", "stroke": "Stroke"}
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# ---------------------------------------- model backend ----------------------------------------
class Backend:
    def __init__(self):
        self.n = len(CLS)
        if args.engine:
            import tensorrt as trt
            self.name = "TensorRT"
            self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(open(args.engine, "rb").read())
            self.ctx = self.engine.create_execution_context(); self.out = torch.empty((1, self.n), dtype=torch.float32, device="cuda")
        else:
            import onnxruntime as ort
            prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
            self.name = "ONNX Runtime (" + prov[0].replace("ExecutionProvider", "") + ")"; self.sess = ort.InferenceSession(args.onnx, providers=prov)
    def __call__(self, x):                          # x: [1,1,96,96,96] float32 torch tensor -> logits numpy [n]
        if args.engine:
            x = x.to("cuda").float().contiguous()
            self.ctx.set_tensor_address("mri", x.data_ptr()); self.ctx.set_tensor_address("logits", self.out.data_ptr())
            assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream), "TensorRT execution failed"
            torch.cuda.synchronize(); return self.out[0].cpu().numpy()
        return self.sess.run(None, {"mri": x.numpy()})[0][0]

def preprocess(path):                               # identical to training: trilinear resize to 96^3, z-score over non-zero voxels
    a = np.nan_to_num(nib.load(path).get_fdata().astype(np.float32))
    if a.ndim == 4: a = a[..., 0]
    x = F.interpolate(torch.from_numpy(a)[None, None], size=(96, 96, 96), mode="trilinear", align_corners=False)
    nz = x != 0; m, s = x[nz].mean(), x[nz].std(unbiased=False)
    return torch.where(nz, (x - m) / (s if s > 0 else 1.0), x).contiguous()

BK = Backend(); BK(torch.zeros(1, 1, 96, 96, 96))   # warm-up so the first click is not slow
print("backend:", BK.name)

SAMPLES = {}
if args.samples:
    for f in sorted(glob.glob(f"{args.samples}/**/*.nii*", recursive=True)): SAMPLES[os.path.basename(f)] = f
TRUE = {}
if args.labels:
    import pandas as pd
    L = pd.read_csv(args.labels); TRUE = {os.path.basename(p): str(l) for p, l in zip(L.path, L.label)}

# ---------------------------------------- inference + figure ----------------------------------------
def run(upload, sample):
    path = None
    if upload: path = upload if isinstance(upload, str) else getattr(upload, "name", None)
    elif sample: path = SAMPLES.get(sample)
    if not path: return "### Choose a scan first", None, ""
    t0 = time.perf_counter(); x = preprocess(path); t1 = time.perf_counter()
    lg = BK(x); t2 = time.perf_counter()
    p = np.exp(lg - lg.max()); p /= p.sum(); k = int(p.argmax()); name = os.path.basename(path)
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.5]}); v = x[0, 0].numpy(); c = v.shape[0] // 2
    for a_, (sl, t) in zip(ax[:3], [(v[:, :, c], "axial"), (v[:, c, :], "coronal"), (v[c, :, :], "sagittal")]):
        a_.imshow(np.rot90(sl), cmap="gray"); a_.set_title(t); a_.axis("off")
    bars = ax[3].barh([NICE.get(c_, c_) for c_ in CLS][::-1], p[::-1], color=[COL[i % 5] for i in range(len(CLS))][::-1])
    for b, val in zip(bars, p[::-1]): ax[3].text(min(val + .02, .98), b.get_y() + b.get_height() / 2, f"{val:.2f}", va="center")
    ax[3].set_xlim(0, 1.1); ax[3].set_title("Class probabilities"); ax[3].spines[["top", "right"]].set_visible(False); fig.tight_layout()
    conf = "" if p[k] >= 0.6 else "  \n**Low confidence**, treat this prediction as uncertain."
    truth = f"  \nTrue label: **{NICE.get(TRUE[name], TRUE[name])}**" + (" (correct)" if TRUE[name] == CLS[k] else " (**wrong**)") if name in TRUE else ""
    head = f"## Prediction: {NICE.get(CLS[k], CLS[k])}  ({p[k]*100:.0f}%)  \nFile: `{name}`{truth}{conf}"
    stats = f"Backend: **{BK.name}** | model inference: **{(t2-t1)*1000:.0f} ms** | preprocessing (resize + normalise): {(t1-t0)*1000:.0f} ms"
    return head, fig, stats

cv = SPEC.get("cv", {}); ws = cv.get("within_source") or {}
about = (f"Fine-tuned BrainIAC (3D ViT-B) on {SPEC['training']['n_scans']} scans {SPEC['training']['per_class']}.  \n"
         f"Cross-validated macro-AUC {cv.get('macro_auc', float('nan')):.2f}, balanced accuracy {cv.get('balanced_accuracy', float('nan')):.2f}."
         + (f"  \nWithin the same-source classes ({', '.join(ws['classes'])}): macro-AUC {ws['macro_auc']:.2f}, balanced accuracy {ws['bal_acc']:.2f}." if ws else "")
         + "  \nStroke and Normal scores partly reflect dataset differences. Inputs must be skull-stripped, atlas-registered NIfTI.")

with gr.Blocks(title="BrainIAC on Jetson") as demo:
    gr.Markdown("# Neuroimaging robot: brain MRI classifier (BrainIAC on Jetson AGX Orin)  \n**Research prototype, not a medical device. Not for diagnosis.**")
    with gr.Row():
        with gr.Column(scale=1):
            up = gr.File(label="Upload a preprocessed scan (.nii / .nii.gz)", file_types=[".nii", ".gz"])
            dd = gr.Dropdown(choices=list(SAMPLES), label="...or pick a demo scan", value=None)
            btn = gr.Button("Run inference", variant="primary")
            with gr.Accordion("About this model", open=False): gr.Markdown(about)
        with gr.Column(scale=2):
            head = gr.Markdown(); fig = gr.Plot(label="Slices and probabilities"); stats = gr.Markdown()
    btn.click(run, [up, dd], [head, fig, stats])

if __name__ == "__main__":
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
