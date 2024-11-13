struct TapeNode
    children::Array{Tuple{Int64, Float64}}  # Child nodes plus the weight that this node contributes
end

mutable struct Expr<:Real
    tape::Base.RefValue{Vector{TapeNode}}
    index::Int64
    value::Float64
end

function add_to_tape_and_return_index(tape_ref::Base.RefValue{Vector{TapeNode}},
                                      children::Array{Tuple{Int64, Float64}})
    new_index = length(tape_ref[]) + 1
    new_node = TapeNode(children)
    push!(tape_ref[], new_node)
    return new_index
end

function Base.:+(x::Expr, y::Expr)
    # Create a new node and add it to the tape
    new_index = add_to_tape_and_return_index(x.tape, [(x.index, 1.0), (y.index, 1.0)])
    # Construct and return a new expression that points to this node
    return Expr(x.tape, new_index, x.value + y.value)
end
function Base.:-(x::Expr, y::Expr)
    new_index = add_to_tape_and_return_index(x.tape, [(x.index, 1.0), (y.index, -1.0)])
    return Expr(x.tape, new_index, x.value - y.value)
end
function Base.:*(x::Expr, y::Expr)
    new_index = add_to_tape_and_return_index(x.tape, [(x.index, y.value), (y.index, x.value)])
    return Expr(x.tape, new_index, x.value * y.value)
end
function Base.sin(x::Expr)
    new_index = add_to_tape_and_return_index(x.tape, [(x.index, cos(x.value))])
    return Expr(x.tape, new_index, sin(x.value))
end
function Base.cos(x::Expr)
    new_index = add_to_tape_and_return_index(x.tape, [(x.index, -sin(x.value))])
    return Expr(x.tape, new_index, cos(x.value))
end

function calculate_derivatives(tape_ref::Base.RefValue{Vector{TapeNode}}, output_index::Int64)
    n_nodes = length(tape_ref[])
    derivs = zeros(n_nodes)
    derivs[output_index] = 1.0
    for i in n_nodes:-1:1
        node = tape_ref[][i]
        for (child_index, weight) in node.children
            derivs[child_index] += weight * derivs[i]
        end
    end
    return derivs
end

function grad_reverse_tape(g, inputs...)
    # Reference to new mpty tape
    tape_ref = Ref(TapeNode[])
    # Forward pass
    exprs = Expr[]
    input_indices = Int64[]
    for i in eachindex(inputs)
        new_index = add_to_tape_and_return_index(tape_ref, Tuple{Int64, Float64}[])
        push!(exprs, Expr(tape_ref, new_index, inputs[i]))
        push!(input_indices, new_index)
    end
    output_expr = g(exprs...)
    # Calculate the derivatives
    derivs = calculate_derivatives(tape_ref, output_expr.index)
    # Get the derivatives we care about
    return output_expr.value, collect(derivs[i] for i in input_indices)
end

f(x, y) = x * y + sin(x)
grad_reverse_tape(f, 1.0, 2.0)
