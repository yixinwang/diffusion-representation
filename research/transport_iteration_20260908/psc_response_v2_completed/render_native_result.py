from pathlib import Path
import numpy as np,json,hashlib
from PIL import Image,ImageDraw
src=Path('work/psc-innovation-response-v2-attempt3/fixed-preview/20260909-response-v2-first16');out=Path('outputs/native-response-v2');out.mkdir(exist_ok=True,parents=True)
bank=np.load(src/'first16.npz');provenance=json.loads((src/'provenance.json').read_text());metrics=json.loads(Path('work/psc-innovation-response-v2-attempt3/independent-cpu-audit/full/20260909-response-v2-independent-audit/audit.json').read_text())
for seed in (78201,78202,78203):
 canvas=Image.new('RGB',(1200,646),'white');d=ImageDraw.Draw(canvas)
 d.text((12,12),f'Native generation study: seed {seed} - first 16 registered samples, all seven arms',fill='black')
 d.text((12,31),'These are generated samples; no quality-based selection, filtering or clipping. The full registered criteria were not met.',fill='black')
 for row,arm in enumerate(('P_frozen','P_joint','I_frozen','I_joint','RQS_frozen','RQS_joint','S42')):
  key=f'{seed}_{arm}';a=bank[key];assert a.shape==(16,3,32,32) and a.dtype==np.float32 and np.isfinite(a).all() and a.min()>=0 and a.max()<=1
  assert hashlib.sha256(a.tobytes()).hexdigest()==provenance['origins'][key]['selected_float32_bytes_sha256']
  top=67+row*78;d.text((10,top+15),arm,fill='black');d.text((10,top+31),f"KID {metrics['seeds'][str(seed)]['metrics'][arm]['kid']:.4f}",fill='black')
  for i,x in enumerate(a):canvas.paste(Image.fromarray(np.rint(x.transpose(1,2,0)*255).astype(np.uint8)).resize((64,64),Image.Resampling.NEAREST),(133+i*66,top))
 d.text((12,626),'Same 3072-D Gaussian row across methods. Display quantization only; full-bank metrics use original arrays.',fill='black')
 canvas.save(out/f'seed_{seed}.png')
print(out)
