using ACEpotentials, AtomsBase, Unitful, AtomsCalculators, JSON

function load_ace(pot_path, add_weights=false)
    model_dict = JSON.parsefile(pot_path)
    model =  ACEpotentials.make_model(model_dict["hyperparams"])

    if add_weights 
        ps2 = deepcopy(model.ps)

        Wpair = convert.(Base.Float64, stack(model_dict["params"]["Wpair"]))
        WB = convert.(Base.Float64, stack(model_dict["params"]["WB"]))

        ps2.Wpair[:] = Wpair
        ps2.WB[:] = WB

        set_parameters!(model, ps2)
    end

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

function eval_observables(atoms, model)
    E, F, V = AtomsCalculators.energy_forces_virial(atoms, model)

    E = Unitful.ustrip(E)
    F = stack(Unitful.ustrip.(F))
    V = stack(Unitful.ustrip.(V))

    return E, F, V
end