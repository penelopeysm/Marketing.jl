function grad_fd(f, inputs...)
    h = 1e-8
    derivs = Float64[]
    orig_value = f(inputs...)
    for i in eachindex(inputs)
        x_plus_h = collect(inputs)
        x_plus_h[i] += h
        x_minus_h = collect(inputs)
        x_minus_h[i] -= h
        deriv = (f(x_plus_h...) - f(x_minus_h...)) / (2 * h)
        push!(derivs, deriv)
    end
    return orig_value, derivs
end
