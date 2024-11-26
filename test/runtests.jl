using Chairmarks
using Logging
using Test
using Marketing

GRAD_FUNCTIONS = [grad_forward, grad_reverse, grad_reverse_tape]

function test_accuracy(grad_function, f, inputs; atol=1e-6)
    value_true = f(inputs...)
    grads_fd = grad_fd(f, inputs...)[2]

    value, grads = grad_function(f, inputs...)
    @test value ≈ value_true atol=atol
    @test grads ≈ grads_fd atol=atol
end

f(x, y) = x * y + sin(x)

FUNCS_AND_INPUTS = [
    (f, [1.0, 2.0])
    (f, [3.0, 5.0])
]

@testset "$(Symbol(f)) @ $(inputs)" for (f, inputs) in FUNCS_AND_INPUTS
    @testset "$grad_function" for grad_function in GRAD_FUNCTIONS
        test_accuracy(grad_function, f, inputs)
    end
    
    @testset "performance $grad_function" for grad_function in [GRAD_FUNCTIONS..., grad_fd]
        res = @b $grad_function(f, inputs...)
        @info "$(Symbol(f)) @ $(inputs) : $(Symbol(grad_function)) $(round(res.time; sigdigits=5))"
    end
end
