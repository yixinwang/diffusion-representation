"""Scalar integer arithmetic only: no model construction, data, or fitting."""
import json
from pathlib import Path
def conv(i,o,k=3):return i*o*k*k+o
def linear(i,o):return i*o+o
def residual(c):return 2*(2*c+conv(c,c))
Hlayer=conv(6,128)+conv(128,128)+conv(128,12,1)
Glayer=conv(6,128)+conv(128,128)+conv(128,150,1)
enc={'stem':conv(3,64),'res64':2*residual(64),'down64_128':conv(64,128,4),'res128_at16':2*residual(128),'down128_128':conv(128,128,4),'res128_at8':2*residual(128),'head':conv(128,16,1)}
dec={'stem':conv(16,128),'res128_at8':2*residual(128),'up128_128':conv(128,128,4),'res128_at16':2*residual(128),'up128_64':conv(128,64,4),'res64_at32':2*residual(64),'head':conv(64,3)}
block=2*(2*128)+2*conv(128,128)+linear(256,256)
time=linear(64,256)+linear(256,256)
def fm(c,blocks):return {'stem':conv(c,128),'time_mlp':time,'residual_blocks':blocks*block,'output_gn':2*128,'output_conv':conv(128,c)}
pixel=fm(3,12);latent=fm(16,8)
r={'assumptions':['every conv, transposed conv, linear has bias','GN has trainable scale+bias','no other trainable embeddings/projections/normalizers','flow conditioner has two3x3 convolutions and1x1 head, noGN','I adds6*23scalar parameters per coupling;16couplings','FM final output hasGN128 then3x3 convolution','fixed64dimtime features carry no trainable parameters'], 'flow':{'H_per_layer':Hlayer,'H':16*Hlayer,'I_extra':16*6*23,'I':16*Hlayer+16*6*23,'G_per_layer':Glayer,'G':16*Glayer},'codec':{'encoder_parts':enc,'encoder':sum(enc.values()),'decoder_parts':dec,'decoder':sum(dec.values()),'total':sum(enc.values())+sum(dec.values())},'fm':{'per_resblock':block,'pixel_parts':pixel,'pixel':sum(pixel.values()),'latent_parts':latent,'latent':sum(latent.values())},'latent_whole_pipeline_parameters':sum(enc.values())+sum(dec.values())+sum(latent.values())}
assert [r['flow'][k] for k in ('H','I','G')]==[2498752,2500960,2783584]
assert [sum(enc.values()),sum(dec.values()),sum(pixel.values()),sum(latent.values())]==[1728272,1744643,4427395,3011472]
Path(__file__).with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
