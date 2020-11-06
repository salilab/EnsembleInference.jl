using EnsembleInference
using Documenter

makedocs(;
    modules=[EnsembleInference],
    authors="Seth Axen <seth.axen@gmail.com> and contributors",
    repo="https://github.com/salilab/EnsembleInference.jl/blob/{commit}{path}#L{line}",
    sitename="EnsembleInference.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://salilab.github.io/EnsembleInference.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/salilab/EnsembleInference.jl",
)
