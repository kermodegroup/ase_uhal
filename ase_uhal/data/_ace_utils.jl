using ACEpotentials, AtomsBase, Unitful, StaticArrays

function load_ace_model(model_json)
    model, meta =  ACEpotentials.load_model(model_json) # load_potential in v0.6 version, load_model in v0.8+
    return model
end

function model_from_params(elements, order, totaldegree, rcut)

    hyperparams = (elements = Symbol.(elements),
        order = order,
        totaldegree = totaldegree,
        rcut = rcut
    )
    return ACEpotentials.ace1_model(; hyperparams...)
end

function convert_ats(atnums, positions, cell, pbc)
    # Simplified version of ASEconvert.ase_to_system 
    # https://github.com/mfherbst/ASEconvert.jl/blob/master/src/ase_conversions.jl
    
    particles = map(1:length(atnums)) do i
        AtomsBase.Atom(AtomsBase.ChemicalSpecies(atnums[i]),
        positions[i, :]u"Å"
        )
    end
    
    cϵll = AtomsBase.PeriodicCell(; cell_vectors=[Vector(cell[i, :]u"Å") for i = 1:3], periodicity=pbc)
    
    return AtomsBase.FlexibleSystem(particles, cϵll)
end

function eval_basis(atoms, model)
    E, F, V = ACEpotentials.Models.energy_forces_virial_basis(atoms, model)

    E = stack(Unitful.ustrip.(E))
    F = stack(Unitful.ustrip.(F))
    V = stack(Unitful.ustrip.(V))

    F = permutedims(F, (3, 2, 1))
    V = permutedims(V, (3, 1, 2))

    return E, F, V
end