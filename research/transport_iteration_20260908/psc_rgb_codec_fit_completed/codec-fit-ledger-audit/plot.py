"""Render preselected first16 FIT originals/reconstructions; never sample/select."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
s=json.loads((a.root/'status.json').read_text());assert s['status']=='completed_FIT_only_qualification'
x=np.load(a.root/'first16_true.npy');y=np.load(a.root/'first16_reconstructed.npy');assert x.shape==y.shape==(16,3,32,32)
assert np.isfinite(x).all() and np.isfinite(y).all() and min(x.min(),y.min())>=0 and max(x.max(),y.max())<=1
q=s['qualification'];canvas=Image.new('RGB',(1168,258),'white');d=ImageDraw.Draw(canvas)
d.text((12,10),'FIT-only codec reconstruction - first 16 registered FIT rows',fill='black')
d.text((12,30),'Training images, not unconditional generation or held-out evaluation.',fill='black')
d.text((12,51),f"Global FIT PSNR: {q['global_FIT_PSNR']:.3f} dB | Mean FIT SSIM: {q['mean_FIT_SSIM']:.4f} | Floors passed: {q['engineering_gates_pass']}",fill='black')
for row,(label,bank) in enumerate([('Original',x),('Reconstruction',y)]):
 top=82+row*78;d.text((8,top+23),label,fill='black')
 for i,img in enumerate(bank):
  arr=np.rint(img.transpose(1,2,0)*255).astype('uint8');canvas.paste(Image.fromarray(arr).resize((64,64),Image.Resampling.NEAREST),(112+i*66,top))
d.text((12,240),'Display quantization only; all reconstruction metrics use saved float32 RGB without clipping.',fill='black')
canvas.save(a.output)
