"""Fabricated-only component checks; no learned quality or real data."""
import copy
import pytest
import torch
from qalt.rgb_codec_flow_matching import RGBCodec,RGBCodecFlowMatching,PixelFlowMatching,ContinuousVelocity

@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1);yield;torch.set_num_threads(old)

def small():return RGBCodecFlowMatching(size=16,latent_channels=2,codec_width=4,field_width=4,codec_blocks=1,field_blocks=2,groups=2)
def images():return torch.rand(3,3,16,16,generator=torch.Generator().manual_seed(811)) * 2-1

def normalized():
    model=small();model.set_stage('normalization');model.update_normalization(images());model.freeze_normalization();return model

def test_default_parameter_counts():
    codec=RGBCodec();pixel=PixelFlowMatching();latent=RGBCodecFlowMatching()
    count=lambda m:sum(p.numel() for p in m.parameters())
    assert count(codec.encoder)==1728272 and count(codec.decoder)==1744643
    assert count(codec)==3472915 and count(pixel)==4427395
    assert count(latent.field)==3011472 and count(latent)==6484387
    assert count(latent.codec.decoder)+count(latent.field)==4756115
    assert latent.source_dimension==1024 and pixel.source_dimension==3072

def test_codec_loss_all_gradients_and_decoder_bounds():
    model=small();x=images();loss=model.training_loss(x);loss.backward()
    assert loss.ndim==0 and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.codec.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.field.parameters())
    y=model.codec(x);assert y.shape==x.shape and y.abs().max()<=1
    with pytest.raises(ValueError):model.codec.encode(x+3)
    with pytest.raises(FloatingPointError):model.codec.decode(torch.full((1,2,4,4),float('inf')))

def test_streaming_moments_reload_and_dtype_preservation():
    model=small().double();x=images().double();model.set_stage('normalization')
    expected=model.codec.encode(x).detach().permute(1,0,2,3).reshape(2,-1)
    model.update_normalization(x[:1]);model.update_normalization(x[1:])
    torch.testing.assert_close(model.moment_mean,expected.mean(1),atol=1e-15,rtol=1e-13)
    model.freeze_normalization();torch.testing.assert_close(model.latent_std,expected.std(1,correction=0))
    assert all(not p.requires_grad for p in model.codec.parameters())
    assert all(p.requires_grad for p in model.field.parameters())
    saved=copy.deepcopy(model.state_dict());clone=small().double();clone.load_state_dict(saved)
    assert torch.equal(clone.latent_mean,model.latent_mean) and torch.equal(clone.latent_std,model.latent_std)
    model.float();assert model.moment_mean.dtype==torch.float64
    assert torch.equal(model.moment_mean,saved['moment_mean'])
    model.reset_normalization();assert model.moment_count==0 and not model.normalization_frozen

def test_latent_flow_gradients_and_stage_guards():
    model=normalized();loss=model.training_loss(images(),torch.Generator().manual_seed(822));loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.field.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.codec.parameters())
    with pytest.raises(ValueError):model.set_stage('codec')
    with pytest.raises(ValueError):model.update_normalization(images())
    model.set_stage('inference')
    with pytest.raises(ValueError):model.training_loss(images())
    pending=small()
    with pytest.raises(ValueError):pending.set_stage('flow')
    with pytest.raises(ValueError):pending.sample_from_gaussian(torch.zeros(1,32))

def test_zero_field_no_hidden_noise_and_checkpoint_copy(monkeypatch):
    model=normalized();model.set_stage('inference')
    with torch.no_grad():
        model.field.output[-1].weight.zero_();model.field.output[-1].bias.zero_()
    clone=small();clone.load_state_dict(copy.deepcopy(model.state_dict()))
    z=torch.randn(2,32,generator=torch.Generator().manual_seed(833))
    expected=model.codec.decode(z.reshape(2,2,4,4)*model.latent_std[None,:,None,None]+model.latent_mean[None,:,None,None])
    def forbidden(*a,**kw):raise AssertionError('sampler requested observed input or random draw')
    monkeypatch.setattr(model.codec,'encode',forbidden);monkeypatch.setattr(torch,'randn',forbidden);monkeypatch.setattr(torch,'rand',forbidden)
    y=model.sample_from_gaussian(z,steps=2)
    assert torch.equal(y,expected) and torch.equal(y,clone.sample_from_gaussian(z,steps=2))
    assert all(not p.requires_grad for p in clone.parameters())
    with pytest.raises(ValueError):model.sample_from_gaussian(z.double(),steps=2)
    with pytest.raises(ValueError):model.sample_from_gaussian(z,steps=True)

def test_pixel_zero_identity_gradients_and_invalids():
    model=PixelFlowMatching(size=16,width=4,blocks=2,groups=2)
    loss=model.training_loss(images(),torch.Generator().manual_seed(844));loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():model.field.output[-1].weight.zero_();model.field.output[-1].bias.zero_()
    model.set_stage('inference');z=torch.randn(1,768,generator=torch.Generator().manual_seed(855))
    assert torch.equal(model.sample_from_gaussian(z,steps=3).flatten(1),z)
    clone=PixelFlowMatching(size=16,width=4,blocks=2,groups=2);clone.load_state_dict(model.state_dict());assert not any(p.requires_grad for p in clone.parameters())
    with pytest.raises(FloatingPointError):model.sample_from_gaussian(z*float('nan'))
    with pytest.raises(ValueError):model.sample_from_gaussian(z[:,:-1])
    with pytest.raises(ValueError):model.field(torch.zeros(1,3,16,16),torch.tensor([1.1]))

def test_degenerate_nonfinite_normalization_rejected():
    model=small();model.set_stage('normalization')
    with pytest.raises(ValueError):model.freeze_normalization()
    with torch.no_grad():
        for p in model.codec.encoder.parameters():p.zero_()
    model.update_normalization(images())
    with pytest.raises(ValueError,match='degenerate'):model.freeze_normalization()
    with pytest.raises(FloatingPointError):model.update_normalization(images()*float('nan'))
    assert not model.normalization_frozen

def test_time_convention_and_nominal_heun_calls():
    model=PixelFlowMatching(size=16,width=4,blocks=1,groups=2).double();model.set_stage('inference')
    times=[]
    handle=model.field.register_forward_hook(lambda m,a,out:times.append(float(a[1][0])))
    with torch.no_grad():model.field.output[-1].weight.zero_();model.field.output[-1].bias.zero_()
    model.sample_from_gaussian(torch.zeros(1,768,dtype=torch.double),steps=2);handle.remove()
    assert times==[0.,.5,.5,1.]
    assert model.field.time_frequencies.shape==(32,)
    assert model.field.time_frequencies[0]==1


def test_cache_loss_matches_image_convenience_and_avoids_codec(monkeypatch):
    model=normalized();x=images();cache=model.encode_normalized(x)
    a=model.training_loss(x,torch.Generator().manual_seed(867))
    def forbidden(*args,**kwargs):raise AssertionError('cache path invoked codec')
    monkeypatch.setattr(model.codec,'encode',forbidden)
    b=model.training_loss_from_latents(cache,torch.Generator().manual_seed(867))
    assert torch.equal(a,b) and not cache.requires_grad
    b.backward();assert all(p.grad is None for p in model.codec.parameters())
    with pytest.raises(ValueError):model.training_loss_from_latents(cache[:,:,:,:-1])
    with pytest.raises(FloatingPointError):model.training_loss_from_latents(cache*float('nan'))
    model.set_stage('inference')
    with pytest.raises(ValueError):model.training_loss_from_latents(cache)


def test_nonzero_field_complete_sampling_reload():
    model=normalized();model.set_stage('inference')
    clone=small();clone.load_state_dict(copy.deepcopy(model.state_dict()))
    z=torch.randn(1,32,generator=torch.Generator().manual_seed(891))
    original=model.sample_from_gaussian(z,steps=2);restored=clone.sample_from_gaussian(z,steps=2)
    assert original.shape==(1,3,16,16) and torch.isfinite(original).all()
    assert original.abs().max()<=1 and torch.equal(original,restored)
    zero_velocity=model.codec.decode(z.reshape(1,2,4,4)*model.latent_std[None,:,None,None]+model.latent_mean[None,:,None,None])
    assert not torch.equal(original,zero_velocity)
