
def install_ace_deps():
    import juliapkg

    juliapkg.add("Unitful")
    juliapkg.add("AtomsBase")
    # juliapkg does not yet seem to support >= specifiers
    # Hack a >= using wildcard version bounds 0.10 - 0.* === [0.10.0, 1.0.0)
    juliapkg.add("ACEpotentials", version="0.10 - 0")

    juliapkg.resolve()