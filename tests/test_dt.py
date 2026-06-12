import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.digital_twin.validators import run_validations

def test_validation_suite():
    assert run_validations()
