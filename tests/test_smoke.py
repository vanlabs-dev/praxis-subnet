import praxis


def test_package_imports() -> None:
    assert praxis is not None


def test_version_exposed() -> None:
    assert hasattr(praxis, "__version__")
    assert isinstance(praxis.__version__, str)
    assert praxis.__version__
