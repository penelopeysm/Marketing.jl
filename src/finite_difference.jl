# Default step size for finite differences. I can't remember the theoretical
# justification for this (or if there's one at all).
const DEFAULT_H = sqrt(eps())

# Chooses the appropriate gradient computation method based on the input/output
# types of the function f.
function grad_fd(f, inputs...; h=DEFAULT_H)
    output = f(inputs...)
    if all(x -> x isa Real, inputs)
        return grad_fd_scalar(f, inputs...; h = h)
    elseif all(x -> x isa AbstractVector{<:Real}, inputs) && output isa AbstractVector{<:Real}
        return grad_fd_vector(f, inputs...; h = h)
    else
        error("Unsupported input/output types")
    end
end

# Differentiate f(a, b, c, ...) = y where:
#   - the inputs (a, b, c, ...) are all scalars
#   - the output y can be a scalar or a vector
#
# Returns a tuple (y, [∂y/∂a, ∂y/∂b, ∂y/∂c, ...])
function grad_fd_scalar(f, inputs...; h=DEFAULT_H)
    y = f(inputs...)
    derivs = typeof(y)[]
    for i in eachindex(inputs)
        x_plus_h = collect(inputs)
        x_plus_h[i] += h
        x_minus_h = collect(inputs)
        x_minus_h[i] -= h
        deriv = (f(x_plus_h...) - f(x_minus_h...)) / (2 * h)
        push!(derivs, deriv)
    end
    return y, derivs
end

# Differentiate f(as, bs, cs, ...) = ys where
#   - the inputs (as, bs, cs, ...) are all vectors
#   - the output ys is a vector
#
# Returns a tuple (ys, [∂ys/∂as, ∂ys/∂bs, ∂ys/∂cs, ...])
# where each ∂ys/∂xs is a Jacobian matrix of size (length(ys), length(xs))
#
# Note that this really corresponds to ReverseDiff.jacobian(f, inputs)
function grad_fd_vector(f, inputs...; h=DEFAULT_H)
    # Run the function once to get the value
    y = f(inputs...)

    # We start by collecting all the inputs into a single vector.
    input_sizes = map(length, inputs)
    inputs_vec = vcat(inputs...)

    # Reconstruct the original inputs based on their known sizes
    function reconstruct_inputs(sizes, vec)
        i = 1
        inputs = []
        for size in sizes
            push!(inputs, vec[i : i+size-1])
            i = i + size
        end
        return inputs
    end
    # Redefine a function f_vec that acts on that input vector. This
    # is the real function that we will differentiate
    function f_vec(inputs_vec)
        f(reconstruct_inputs(input_sizes, inputs_vec)...)
    end

    # Assume f_vec : R^n -> R^m (where n is the total number of inputs).
    # We want to return an m * n Jacobian matrix where J_{ij} = dy_i/dx_j
    # We can calculate one column of the Jacobian by:
    #    Δy = (f_vec(x + h e_j) - f_vec(x)) / 2h
    # This returns a vector of length m which is the j-th
    # column of the Jacobian
    n = length(inputs_vec)
    m = length(y)
    J = zeros(m, n)
    for j in 1:n
        x_plus_h = copy(inputs_vec)
        x_plus_h[j] += h
        x_minus_h = copy(inputs_vec)
        x_minus_h[j] -= h
        dy_j = (f_vec(x_plus_h) - f_vec(x_minus_h)) / (2 * h)
        J[:, j] = dy_j
    end

    # We need to split the Jacobian up for each input. Each input of
    # size s_i would get s_i columns' worth of the Jacobian
    function reconstruct_jacobians(sizes, J)
        i = 1
        jacs = Matrix{Float64}[]
        for size in sizes
            push!(jacs, J[:, i : i+size-1])
            i = i + size
        end
        return jacs
    end

    return y, reconstruct_jacobians(input_sizes, J)
end
