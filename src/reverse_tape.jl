mutable struct TapeNode<:Real
    tape::Base.RefValue{Vector{TapeNode}}
    value::Float64
    deriv::Union{Float64,Nothing}
    children::Array{Tuple{Int64, Float64}}  # Child nodes plus the weight that this node contributes
end

function Base.:+(x::TapeNode, y::TapeNode)
    tape = x.tape
    new_node = TapeNode(tape, x.value + y.value, nothing, [])
    new_index = length(tape[]) + 1
    push!(tape[], new_node)
    push!(x.children, (new_index, 1.0))  # 1.0 = d(x + y)/dx
    push!(y.children, (new_index, 1.0))  # 1.0 = d(x + y)/dy
    return new_node
end
function Base.:-(x::TapeNode, y::TapeNode)
    tape = x.tape
    new_node = TapeNode(tape, x.value - y.value, nothing, [])
    new_index = length(tape[]) + 1
    push!(tape[], new_node)
    push!(x.children, (new_index, 1.0))   # 1.0 = d(x - y)/dx
    push!(y.children, (new_index, -1.0))  # -1.0 = d(x - y)/dy
    return new_node
end
function Base.:*(x::TapeNode, y::TapeNode)
    tape = x.tape
    new_node = TapeNode(tape, x.value * y.value, nothing, [])
    new_index = length(tape[]) + 1
    push!(tape[], new_node)
    push!(x.children, (new_index, y.value))  # y.value = d(x * y)/dx
    push!(y.children, (new_index, x.value))  # x.value = d(x * y)/dy
    return new_node
end
function Base.sin(x::TapeNode)
    tape = x.tape
    new_node = TapeNode(tape, sin(x.value), nothing, [])
    new_index = length(tape[]) + 1
    push!(tape[], new_node)
    push!(x.children, (new_index, cos(x.value)))  # cos(x.value) = d(sin(x))/dx
    return new_node
end
function Base.cos(x::TapeNode)
    tape = x.tape
    new_node = TapeNode(tape, cos(x.value), nothing, [])
    new_index = length(tape[]) + 1
    push!(tape[], new_node)
    push!(x.children, (new_index, -sin(x.value)))  # -sin(x.value) = d(cos(x))/dx
    return new_node
end

function calc_derivative!(node::TapeNode)
    if isnothing(node.deriv)
        node.deriv = sum([
            weight * calc_derivative!(node.tape[][child_index])
            for (child_index, weight) in node.children
        ])
    end
    return node.deriv
end

function grad_reverse_tape(g, inputs...)
    # Empty tape
    tape = TapeNode[]
    # Forward pass
    node_inputs = [TapeNode(Ref(tape), input, nothing, []) for input in inputs]
    output_node = g(node_inputs...)
    # Calculate the derivatives
    output_node.deriv = 1.0
    derivs = [calc_derivative!(node) for node in node_inputs]
    # Return the output and the derivatives
    return output_node.value, derivs
end
