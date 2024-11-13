using Test
using Marketing

TEST_FUNCTIONS = [grad_forward, grad_reverse, grad_reverse_tape]

function test_accuracy(grad_function, f, inputs; atol=1e-6)
    value_true = f(inputs...)
    grads_fd = grad_fd(f, inputs...)[2]

    value, grads = grad_function(f, inputs...)
    @test value ≈ value_true atol=atol
    @test grads ≈ grads_fd atol=atol
end

@testset "f(x,y)=x*y+sin(x)" begin
    for grad_function in TEST_FUNCTIONS
        f(x, y) = x*y + sin(x)
        inputs = [1.0, 2.0]
        test_accuracy(grad_function, f, inputs)
    end
end
