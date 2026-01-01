"""Basic sanity tests."""


def test_sanity():
    """Basic sanity check."""
    assert True


def test_imports():
    """Test that main modules can be imported."""
    import qvl
    from qvl import config, runner, artifacts, plotting
    from qvl.batteries import base
    from qvl.backends import toy

    assert qvl.__version__ == "0.1.0"
