"""Pinned upstream source and local-only weights. No network fallbacks."""
import hashlib
import importlib.util
from pathlib import Path
import numpy as np

SOURCE_SHA256 = 'c6183fff54dd240fe66d53d207f4bd28c06fde98c21b5525f10ca0cc5cef7780'
WEIGHT_SHA256 = '6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'
WEIGHT_BYTES = 95628359
WEIGHT_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_artifacts(source, weights):
    if digest(source) != SOURCE_SHA256:
        raise ValueError('upstream source hash mismatch')
    if Path(weights).stat().st_size != WEIGHT_BYTES or digest(weights) != WEIGHT_SHA256:
        raise ValueError('upstream weights hash/size mismatch')


def make_extractor(source, weights, device):
    import torch
    validate_artifacts(source, weights)
    source_bytes = Path(source).read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != SOURCE_SHA256:
        raise ValueError('upstream source changed before import')
    spec = importlib.util.spec_from_file_location('pinned_fid_inception', source)
    upstream = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes,str(source),'exec'),upstream.__dict__)

    def local_state(url, *args, **kwargs):
        if url != WEIGHT_URL or args or set(kwargs)-{'progress'}:
            raise ValueError('unexpected upstream loader request')
        # Read and hash the exact same bytes supplied to the restricted loader.
        import io
        payload = Path(weights).read_bytes()
        if len(payload) != WEIGHT_BYTES or hashlib.sha256(payload).hexdigest() != WEIGHT_SHA256:
            raise ValueError('weight bytes changed')
        return torch.load(io.BytesIO(payload), weights_only=True, map_location='cpu')

    upstream.load_state_dict_from_url = local_state
    model = upstream.InceptionV3(output_blocks=(3,), resize_input=True,
                                 normalize_input=True, requires_grad=False,
                                 use_fid_inception=True)
    model.eval().to(device=device, dtype=torch.float32)
    return model


def extract(model, images, device):
    import torch
    if images.dtype != np.float64 or images.ndim != 4 or images.shape[1:] != (3,32,32):
        raise ValueError('float64 native RGB NCHW required')
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError('image outside finite closed unit cube; clipping forbidden')
    chunks = []
    with torch.inference_mode():
        for first in range(0,len(images),32):
            batch = np.array(images[first:first+32],dtype=np.float32,copy=True)
            features = model(torch.from_numpy(batch).to(device))[0].flatten(1)
            if features.shape[1] != 2048 or not bool(torch.isfinite(features).all()):
                raise FloatingPointError('invalid features')
            chunks.append(features.cpu().numpy())
    return np.concatenate(chunks)
