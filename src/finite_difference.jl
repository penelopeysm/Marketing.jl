function grad_fd(f, inputs...)
    h = 1e-10
    derivs = Float64[]
    orig_value = f(inputs...)
    for i in eachindex(inputs)
        x_plus_h = collect(inputs)
        x_plus_h[i] += h
        deriv = (f(x_plus_h...) - orig_value) / h
        push!(derivs, deriv)
    end
    return orig_value, derivs
end
