"""FragFM: Fragment-based discrete flow matching for molecular generation."""

__all__ = ["FragFMGenerator"]


def __getattr__(name: str):
    if name == "FragFMGenerator":
        from mf_generators.fragfm.generator import FragFMGenerator

        return FragFMGenerator
    raise AttributeError(name)
