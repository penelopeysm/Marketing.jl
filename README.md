Marketing.jl
------------

Toy implementation of automatic differentiation using operator overloading.

### Usage

```julia
julia> using Marketing

julia> f(x, y) = x * y + sin(x)
f (generic function with 1 method)

julia> grad_forward(f, 1.0, 2.0)   # Forward-mode
(2.8414709848078967, [2.5403023058681398, 1.0])

julia> grad_reverse(f, 1.0, 2.0)   # Reverse-mode
(2.8414709848078967, [2.5403023058681398, 1.0])

julia> grad_reverse_tape(f, 1.0, 2.0)   # Reverse-mode using a tape
(2.8414709848078967, [2.5403023058681398, 1.0])

julia> grad_fd(f, 1.0, 2.0)   # Finite differences (central)
(2.8414709848078967, [2.5403023240500033, 0.999999993922529])
```
