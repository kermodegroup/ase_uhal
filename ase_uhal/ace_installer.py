
def install_ace_deps():
    try:
        import juliapkg
    except ImportError as e:
        err_text = ("ase_uhal is not installed with ACE compatibility.\n"
                  + " "*13 # Indentation
                  + "Run `pip install ase_uhal[ace]` to resolve"
        )
        raise ImportError(err_text) from e
    juliapkg.add("Unitful")
    juliapkg.add("AtomsBase")
    # juliapkg does not yet seem to support >= specifiers
    # Hack a >= using wildcard version bounds 0.10 - 0.* === [0.10.0, 1.0.0)
    juliapkg.add("ACEpotentials", version="0.10 - 0")

    juliapkg.resolve()