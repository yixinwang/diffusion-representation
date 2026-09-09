import copy

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from qalt.learned_latent_flow_matching import (
    GlobalConditionalVelocity,
    LearnedLatentFlowMatching,
    pack_residuals,
    unpack_residuals,
)
from qalt.multiscale_flow import haar_split


@pytest.fixture(scope="module", autouse=True)
def _small_cpu_tests():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _small(**options):
    config = dict(channels=1, size=8, levels=2, pre_layers=2, coarse_layers=2,
                  detail_layers=2, width=8, bins=4, attention_heads=2)
    config.update(options)
    return LearnedLatentFlowMatching(**config).double()


def test_residual_packing_is_a_dense_jacobian_permutation_and_exact_inverse():
    source = torch.arange(15, dtype=torch.float64, requires_grad=True)

    def packed_map(vector):
        return pack_residuals((vector[:3].reshape(1, 3, 1, 1),
                               vector[3:].reshape(1, 3, 2, 2))).flatten()

    jacobian = torch.autograd.functional.jacobian(packed_map, source)
    assert jacobian.shape == (15, 15)
    assert torch.equal(jacobian.sum(0), torch.ones(15, dtype=source.dtype))
    assert torch.equal(jacobian.sum(1), torch.ones(15, dtype=source.dtype))
    assert torch.all((jacobian == 0) | (jacobian == 1))
    _, determinant = torch.linalg.slogdet(jacobian)
    assert determinant == 0
    unpacked = unpack_residuals(packed_map(source).reshape(1, 15, 1, 1), 1, 2)
    assert torch.equal(torch.cat([block.flatten() for block in unpacked]), source)
    with pytest.raises(ValueError, match="channel count"):
        unpack_residuals(torch.zeros(1, 14, 2, 2), 1, 2)
    with pytest.raises(ValueError, match="dyadic"):
        pack_residuals((torch.zeros(1, 3, 2, 2), torch.zeros(1, 3, 3, 3)))


@pytest.mark.parametrize("channels,size,latent,residual_channels", [(1, 8, 4, 15), (3, 32, 192, 45)])
def test_full_gaussian_source_budget_and_parameter_accounting(channels, size, latent, residual_channels):
    model = _small(channels=channels, size=size)
    assert model.dimension == channels * size * size
    assert model.latent_dimension == latent
    assert model.packed_residual_channels == residual_channels
    assert model.residual_dimension == model.dimension - latent
    assert residual_channels * model.coarse_size**2 == model.residual_dimension
    counts = model.parameter_counts
    assert counts["analysis"] > 0 and counts["coarse_fm"] > 0 and counts["residual_fm"] > 0
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert counts["total"] == counts["analysis"] + counts["coarse_fm"] + counts["residual_fm"]


@pytest.mark.parametrize("unit_interval", [False, True])
def test_composed_analysis_roundtrip_and_logdet_including_prehaar(unit_interval):
    torch.manual_seed(571)
    model = _small(unit_interval=unit_interval)
    values = torch.randn(3, 1, 8, 8, dtype=torch.float64) * 0.3
    if unit_interval:
        values = torch.sigmoid(values)
    encoded, determinant = model.encode_analysis(values)
    recovered, inverse = model.decode_analysis(encoded)
    torch.testing.assert_close(recovered, values, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(determinant + inverse, torch.zeros_like(inverse), atol=2e-12, rtol=0)
    assert encoded.shape == (3, 64)


def test_learned_prehaar_mixing_can_move_original_detail_information_into_code():
    torch.manual_seed(651)
    model = _small()
    # A nonzero conditional parameter map represents an available learned state;
    # no optimization or empirical-quality experiment is performed in this test.
    with torch.no_grad():
        for layer in model.pre_analysis.layers:
            layer.net[-1].weight.normal_(std=0.15)
    image = 0.2 * torch.randn(1, 1, 8, 8, dtype=torch.float64)
    perturbation = torch.zeros_like(image)
    perturbation[0, 0, 0, 0] = 0.15
    perturbation[0, 0, 0, 1] = -0.15
    original_coarse = image
    changed_coarse = image + perturbation
    for _ in range(2):
        original_coarse, _ = haar_split(original_coarse)
        changed_coarse, _ = haar_split(changed_coarse)
    torch.testing.assert_close(original_coarse, changed_coarse, atol=1e-15, rtol=0)
    first, _ = model.encode_analysis(image)
    second, _ = model.encode_analysis(image + perturbation)
    assert (first[:, :4] - second[:, :4]).abs().max() > 1e-7


def test_stage_freezing_blocks_analysis_and_input_gradients_and_survives_loading():
    torch.manual_seed(771)
    model = _small()
    observations = torch.randn(2, 1, 8, 8, dtype=torch.float64) * 0.2
    with pytest.raises(RuntimeError, match="freeze_analysis"):
        model.training_loss(observations)
    with pytest.raises(RuntimeError, match="freeze_analysis"):
        model.sample_from_gaussian(torch.zeros(2, 64, dtype=torch.float64))
    model.train_analysis_loss(observations).backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.pre_analysis.parameters())
    assert all(p.grad is None for p in model.coarse_prior.parameters())
    assert all(p.grad is None for p in model.residual_velocity.parameters())
    model.freeze_analysis()
    model.train()
    assert not model.analysis.training and not model.pre_analysis.training
    assert all(not p.requires_grad and p.grad is None for p in model._analysis_parameters())
    observations.requires_grad_()
    loss = model.training_loss(observations, torch.Generator().manual_seed(227))
    loss.backward()
    assert observations.grad is None
    assert all(p.grad is None for p in model._analysis_parameters())
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.coarse_prior.parameters())
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.residual_velocity.parameters())
    with pytest.raises(RuntimeError, match="frozen"):
        model.train_analysis_loss(observations)
    restored = _small()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    assert restored.analysis_frozen
    assert all(not p.requires_grad for p in restored._analysis_parameters())
    restored.train()
    assert not restored.analysis.training
    source = torch.randn(2, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(744)) * 0.2
    torch.testing.assert_close(model.sample_from_gaussian(source, steps=1),
                               restored.sample_from_gaussian(source, steps=1), atol=0, rtol=0)


def test_zero_field_training_loss_weights_every_encoded_coordinate_equally():
    model = _small()
    model.freeze_analysis()
    observations = torch.randn(2, 1, 8, 8, dtype=torch.float64,
                               generator=torch.Generator().manual_seed(412)) * 0.2
    code, _ = model.encode_analysis(observations)
    coarse, residual = model._split_code(code)
    target = torch.cat((coarse.flatten(1), residual.flatten(1)), dim=1)
    source = torch.randn(2, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(718))
    loss = model.training_loss(observations, torch.Generator().manual_seed(718))
    torch.testing.assert_close(loss, (target - source).square().mean(), atol=3e-15, rtol=3e-15)


def test_global_field_has_direct_distant_state_and_context_dependence():
    torch.manual_seed(144)
    field = GlobalConditionalVelocity(3, 1, size=8, width=8, heads=2).double()
    with torch.no_grad():
        field.output.weight.normal_(std=0.2)
    state = torch.randn(1, 3, 8, 8, dtype=torch.float64, requires_grad=True)
    context = torch.randn(1, 1, 8, 8, dtype=torch.float64, requires_grad=True)
    value = field(state, torch.tensor([0.4], dtype=torch.float64), context)[0, 0, 0, 0]
    state_gradient, context_gradient = torch.autograd.grad(value, (state, context))
    # These locations are outside the convolution-only receptive field.
    assert state_gradient[0, :, -1, -1].abs().max() > 1e-9
    assert context_gradient[0, :, -1, -1].abs().max() > 1e-9


class _UnitVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, t, context=None):
        assert context is None
        self.calls += 1
        return torch.ones_like(x)


class _ContextRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.contexts = []

    def forward(self, x, t, context=None):
        assert context is not None
        self.contexts.append(context.clone())
        return context.repeat(1, x.shape[1] // context.shape[1], 1, 1)


def test_generation_uses_only_generated_coarse_context_and_accepts_no_teacher():
    model = _small()
    model.freeze_analysis()
    model.coarse_prior.velocity = _UnitVelocity()
    model.residual_velocity = _ContextRecorder()
    source = torch.zeros(1, 64, dtype=torch.float64)
    before = model.sample_from_gaussian(source, steps=3)
    assert len(model.residual_velocity.contexts) == 6
    assert model.coarse_prior.velocity.calls == 6
    assert all(torch.equal(c, torch.ones_like(c)) for c in model.residual_velocity.contexts)
    expected_code = model._join_code(torch.ones(1, 1, 2, 2, dtype=torch.float64),
                                     torch.ones(1, 15, 2, 2, dtype=torch.float64))
    expected, _ = model.decode_analysis(expected_code)
    torch.testing.assert_close(before, expected, atol=2e-13, rtol=2e-13)
    model.training_loss(torch.full((1, 1, 8, 8), 4.0, dtype=torch.float64))
    model.residual_velocity.contexts.clear()
    after = model.sample_from_gaussian(source, steps=3)
    torch.testing.assert_close(after, before, atol=0, rtol=0)
    assert all(torch.equal(c, torch.ones_like(c)) for c in model.residual_velocity.contexts)
    with pytest.raises(TypeError):
        model.sample_from_gaussian(source, context=torch.ones(1, 1, 2, 2))


def test_zero_fields_retain_all_source_coordinates_through_full_analysis_inverse():
    model = _small()
    model.freeze_analysis()
    source = torch.randn(2, 64, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(484)) * 0.3
    observations = model.sample_from_gaussian(source, steps=2)
    code, _ = model.encode_analysis(observations)
    coarse, residual = model._split_code(code)
    recovered = torch.cat((coarse.flatten(1), residual.flatten(1)), dim=1)
    torch.testing.assert_close(recovered, source, atol=2e-13, rtol=2e-13)
    with pytest.raises(ValueError, match="every Gaussian coordinate"):
        model.sample_from_gaussian(source[:, :-1])
    with pytest.raises(ValueError, match="positive integer"):
        model.sample_from_gaussian(source, steps=0)
    with pytest.raises(ValueError, match="at least two"):
        _small(pre_layers=1)
    with pytest.raises(ValueError, match="divisible"):
        _small(width=7)


def test_analysis_likelihood_is_not_assigned_to_the_composed_fm_distribution():
    model = _small()
    observations = torch.full((1, 1, 8, 8), 0.2, dtype=torch.float64)
    analysis_loss = model.train_analysis_loss(observations).detach()
    # A constant coarse velocity changes the final model law without changing
    # its separately fitted analysis likelihood.
    with torch.no_grad():
        model.coarse_prior.velocity.net[-1].bias.fill_(0.25)
    torch.testing.assert_close(model.train_analysis_loss(observations), analysis_loss,
                               atol=0, rtol=0)
    assert not hasattr(model, "log_prob")
    model.freeze_analysis()
    source = torch.zeros(1, 64, dtype=torch.float64)
    generated = model.sample_from_gaussian(source, steps=2)
    code, _ = model.encode_analysis(generated)
    torch.testing.assert_close(code[:, :model.latent_dimension],
                               torch.full((1, 4), 0.25, dtype=torch.float64),
                               atol=2e-13, rtol=2e-13)
