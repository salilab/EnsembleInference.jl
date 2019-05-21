using Documenter, EnsembleInference

makedocs(
    modules = [EnsembleInference],
    format = :html,
    checkdocs = :exports,
    sitename = "EnsembleInference.jl",
    pages = Any["index.md"]
)

deploydocs(
    repo = "github.com/sdaxen/EnsembleInference.jl.git",
)
