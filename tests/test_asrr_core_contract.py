import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class ASRRCoreContractTests(unittest.TestCase):
    def test_action_only_adapter_starts_as_identity_residual(self):
        import torch
        from asrr_core import ActionSequenceResidualAdapter

        adapter = ActionSequenceResidualAdapter(
            action_dim=7,
            horizon=5,
            hidden_dim=32,
            fusion_mode="action_only",
            head_type="dense",
        )
        base_action = torch.randn(2, 5, 7)
        delta = adapter(base_action)

        self.assertEqual(tuple(delta.shape), (2, 5, 7))
        self.assertTrue(torch.allclose(delta, torch.zeros_like(delta)))

    def test_supervised_loss_backpropagates(self):
        import torch
        from asrr_core import ActionSequenceResidualAdapter, supervised_asrr_loss

        adapter = ActionSequenceResidualAdapter(
            action_dim=4,
            horizon=6,
            state_context_dim=3,
            hidden_dim=32,
            fusion_mode="state_add",
            head_type="bounded_dense",
            max_delta=0.1,
        )
        base_action = torch.randn(2, 6, 4)
        state_context = torch.randn(2, 3)
        target_action = torch.randn(2, 6, 4)

        delta = adapter(base_action, state_context=state_context)
        metrics = supervised_asrr_loss(
            base_action=base_action,
            delta_action=delta,
            target_action=target_action,
            loss_type="mse",
        )
        metrics["loss"].backward()

        self.assertIn("refined_loss", metrics)
        self.assertTrue(torch.isfinite(metrics["loss"]))

    def test_coordinate_mask_is_applied_by_refiner(self):
        import torch
        from asrr_core import ActionSequenceResidualAdapter

        adapter = ActionSequenceResidualAdapter(
            action_dim=3,
            horizon=4,
            hidden_dim=16,
            fusion_mode="action_only",
            head_type="bounded_dense",
            max_delta=0.1,
            action_mask=[1.0, 0.0, 1.0],
        )
        with torch.no_grad():
            adapter.delta_head.bias.fill_(1.0)
        delta = adapter(torch.zeros(2, 4, 3))

        self.assertTrue(torch.all(delta[..., 1] == 0))
        self.assertTrue(torch.all(delta[..., (0, 2)] > 0))

    def test_padding_does_not_dilute_reconstruction_loss(self):
        import torch
        from asrr_core import supervised_asrr_loss

        base = torch.zeros(1, 2, 1)
        target = torch.ones(1, 2, 1)
        metrics = supervised_asrr_loss(
            base_action=base,
            delta_action=torch.zeros_like(base),
            target_action=target,
            is_pad=torch.tensor([[False, True]]),
            delta_l2_weight=0.0,
            loss_type="l1",
        )

        self.assertAlmostEqual(float(metrics["refined_loss"]), 1.0)

    def test_cached_dataset_separates_global_action_mask(self):
        import torch
        from asrr_core import CachedActionDataset

        dataset = CachedActionDataset(
            {
                "base_action": torch.zeros(3, 4, 2),
                "target_action": torch.ones(3, 4, 2),
                "action_mask": torch.tensor([1.0, 0.0]),
            }
        )

        self.assertEqual(len(dataset), 3)
        self.assertEqual(set(dataset[0]), {"base_action", "target_action"})
        self.assertTrue(torch.equal(dataset.action_mask, torch.tensor([1.0, 0.0])))

    def test_pi05_wrapper_preserves_shape_and_identity_start(self):
        import torch
        from examples.pi05 import Pi05ASRRWrapper, build_pi05_refiner

        base = torch.randn(2, 5, 6)
        wrapper = Pi05ASRRWrapper(
            lambda: {"base_action": base},
            build_pi05_refiner(action_dim=6, horizon=5, hidden_dim=16),
            device="cpu",
        )
        output = wrapper.predict_action_chunk()

        self.assertEqual(tuple(output["action"].shape), (2, 5, 6))
        self.assertTrue(torch.allclose(output["action"], base))

    def test_diffusion_wrapper_keeps_base_in_eval_mode(self):
        import torch
        import torch.nn as nn
        from examples.diffusion_policy.asrr_dp_wrapper import (
            ASRRDiffusionPolicyWrapper,
            DPASRRAdapter,
        )

        class FakePolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))

            def predict_action(self, obs_dict):
                batch = obs_dict["obs"].shape[0]
                return {"action": torch.zeros(batch, 4, 2)}

        base = FakePolicy()
        wrapper = ASRRDiffusionPolicyWrapper(
            base,
            DPASRRAdapter(action_dim=2, horizon=4, asrr_variant="asrr_action", hidden_dim=16),
        )
        wrapper.train()
        output = wrapper.predict_action({"obs": torch.zeros(3, 2, 5)})

        self.assertFalse(base.training)
        self.assertEqual(tuple(output["action"].shape), (3, 4, 2))
        self.assertTrue(torch.allclose(output["action"], output["base_action"]))


if __name__ == "__main__":
    unittest.main()
