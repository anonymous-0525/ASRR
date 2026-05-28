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


if __name__ == "__main__":
    unittest.main()
