module Marketing

export grad_forward, grad_reverse, grad_reverse_tape, grad_fd

include("forward.jl")
include("reverse.jl")
include("reverse_tape.jl")
include("finite_difference.jl")

end
