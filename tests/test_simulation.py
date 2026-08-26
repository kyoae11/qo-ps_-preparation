"""仿真数据核心不变量测试。"""

import json
import unittest
from pathlib import Path

import numpy as np

from afil.paths import PROJECT_ROOT
from qops_preparation.simulation import (
    build_entities,
    build_entity_feature_table,
    build_service_mapping,
    load_config,
    perturb_matrices,
)


class SimulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(PROJECT_ROOT / "config" / "simulation_v1.json")
        cls.static, cls.dynamic, cls.users = build_entities(cls.config)
        cls.mapping = build_service_mapping(4612, cls.static, cls.dynamic, cls.config["seed"])
        cls.features = build_entity_feature_table(cls.static, cls.dynamic, cls.mapping)

    def test_counts_and_mapping(self):
        self.assertEqual(len(self.static), 192)
        self.assertEqual(len(self.dynamic), 64)
        self.assertEqual(len(self.users), 339)
        self.assertEqual(len(self.mapping), 4612)
        self.assertEqual(self.mapping["service_id"].nunique(), 4612)
        self.assertEqual(self.mapping["entity_id"].nunique(), 256)

    def test_generation_is_deterministic(self):
        static2, dynamic2, users2 = build_entities(self.config)
        np.testing.assert_allclose(self.static["processing_capacity"], static2["processing_capacity"])
        np.testing.assert_allclose(self.dynamic["device_load"], dynamic2["device_load"])
        np.testing.assert_allclose(self.users["location_x_m"], users2["location_x_m"])

    def test_perturbation_direction_and_no_noise(self):
        rt = np.array([[1.0, 2.0], [-1.0, 0.5]])
        tp = np.array([[10.0, 20.0], [-1.0, 5.0]])
        features = self.features.iloc[:2].reset_index(drop=True)
        rt_ps, tp_ps, factors = perturb_matrices(rt, tp, features, self.config)
        self.assertTrue(np.all(rt_ps[rt > 0] >= rt[rt > 0]))
        self.assertTrue(np.all(tp_ps[tp > 0] <= tp[tp > 0]))
        self.assertEqual(rt_ps[1, 0], -1.0)
        self.assertEqual(tp_ps[1, 0], -1.0)
        expected = rt * factors["rt_multiplier"].to_numpy()[None, :]
        np.testing.assert_allclose(rt_ps[rt > 0], expected[rt > 0])

    def test_zero_strength_is_identity(self):
        config = json.loads(json.dumps(self.config))
        config["perturbation"]["lambda_rt"] = 0.0
        config["perturbation"]["lambda_tp"] = 0.0
        rt = np.array([[1.0, -1.0]])
        tp = np.array([[10.0, -1.0]])
        rt_ps, tp_ps, _ = perturb_matrices(rt, tp, self.features.iloc[:2], config)
        np.testing.assert_allclose(rt_ps, rt)
        np.testing.assert_allclose(tp_ps, tp)


if __name__ == "__main__":
    unittest.main()
