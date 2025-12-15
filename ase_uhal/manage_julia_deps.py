import juliapkg

jl_json = "../juliapkg.json"

juliapkg.add("ACEpotentials", target=jl_json)
juliapkg.add("Unitful", target=jl_json)
juliapkg.add("AtomsBase", target=jl_json)
