"""
Shared pytest fixtures for happy-simulator tests.
"""

import pytest
from pathlib import Path
from datetime import datetime


@pytest.fixture(scope="session")
def test_output_root() -> Path:
    """
    Returns the root test_output directory. Created once per test session.
    Files here persist after tests complete for easy access.
    """
    output_dir = Path(__file__).parent.parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def test_output_dir(request, test_output_root) -> Path:
    """
    Returns a directory for the current test to write output files.
    Directory structure: test_output/<module_name>/<test_name>/
    
    Example usage:
        def test_visualization(test_output_dir):
            import matplotlib.pyplot as plt
            plt.plot([1, 2, 3], [1, 4, 9])
            plt.savefig(test_output_dir / "my_plot.png")
            
            # Also save raw data
            with open(test_output_dir / "data.csv", "w") as f:
                f.write("x,y\\n1,1\\n2,4\\n3,9\\n")
    """
    # Get module name (e.g., "test_simulation_basic_counter")
    module_name = request.module.__name__.split(".")[-1]
    # Get test function name (e.g., "test_counter_increments")
    test_name = request.node.name
    
    test_dir = test_output_root / module_name / test_name
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
def timestamped_output_dir(request, test_output_root) -> Path:
    """
    Like test_output_dir but includes a timestamp, useful when you want to
    keep multiple runs of the same test for comparison.
    
    Directory structure: test_output/<module>/<test>/<timestamp>/
    """
    module_name = request.module.__name__.split(".")[-1]
    test_name = request.node.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    test_dir = test_output_root / module_name / test_name / timestamp
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir
