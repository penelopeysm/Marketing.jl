struct Dual<:Real
    value::Float64
    deriv::Float64
end

Base.:+(x::Dual, y::Dual) = Dual(x.value + y.value, x.deriv + y.deriv)
Base.:-(x::Dual, y::Dual) = Dual(x.value - y.value, x.deriv - y.deriv)
Base.:*(x::Dual, y::Dual) = Dual(x.value * y.value, x.deriv * y.value + x.value * y.deriv)
Base.sin(x::Dual) = Dual(sin(x.value), cos(x.value) * x.deriv)
Base.cos(x::Dual) = Dual(cos(x.value), -sin(x.value) * x.deriv)

function grad_forward(f, inputs...)
    derivs = Float64[]
    output_value = nothing
    for i in eachindex(inputs)
        duals = [Dual(inputs[j], i == j ? 1.0 : 0.0)   
            for j in eachindex(inputs)]
        push!(derivs, f(duals...).deriv)
        # Set the value too
        output_value = f(duals...).value
    end
    return output_value, derivs
end
