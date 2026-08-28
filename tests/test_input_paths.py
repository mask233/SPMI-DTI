from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from config import hyperparameter


class InputPathTests(unittest.TestCase):
    def test_default_layout_is_backward_compatible(self):
        with patch.dict(os.environ, {}, clear=True):
            hp = hyperparameter("Davis")
        self.assertEqual(hp.data_dir, os.path.normpath("./DataSets"))
        self.assertEqual(hp.preprocess_dir, os.path.normpath("./DataSets/Preprocessed"))
        self.assertEqual(hp.pdb_dir, os.path.normpath("./DataSets/pdb/Davis_pdb"))
        self.assertEqual(hp.subpocket_dir, os.path.normpath("./DataSets/pdb/Davis_subpocket"))

    def test_environment_layout(self):
        environment = {
            "DTI_DATA_DIR": "/srv/dti/input",
            "DTI_PREPROCESS_DIR": "/srv/dti/cache",
        }
        with patch.dict(os.environ, environment, clear=True):
            hp = hyperparameter("DrugBank")
        self.assertEqual(hp.data_dir, os.path.normpath("/srv/dti/input"))
        self.assertEqual(hp.preprocess_dir, os.path.normpath("/srv/dti/cache"))
        self.assertEqual(hp.pdb_root, os.path.normpath("/srv/dti/input/pdb"))

    def test_explicit_directories_override_environment(self):
        environment = {
            "DTI_DATA_DIR": "/wrong/data",
            "DTI_PREPROCESS_DIR": "/wrong/cache",
        }
        with patch.dict(os.environ, environment, clear=True):
            hp = hyperparameter(
                "BIOSNAP", data_dir="/chosen/data", preprocess_dir="/chosen/cache"
            )
        self.assertEqual(hp.data_dir, os.path.normpath("/chosen/data"))
        self.assertEqual(hp.preprocess_dir, os.path.normpath("/chosen/cache"))


if __name__ == "__main__":
    unittest.main()
