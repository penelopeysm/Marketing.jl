mutable struct Node<:Real
    value::Float64
    deriv::Union{Float64,Nothing}
    children::Array{Tuple{Node, Float64}}  # Child nodes plus the weight that this node contributes
end

# Note that all of these mutate the input nodes
function Base.:+(x::Node, y::Node)
    new_node = Node(x.value + y.value, nothing, [])
    push!(x.children, (new_node, 1.0))  # 1.0 = d(x + y)/dx
    push!(y.children, (new_node, 1.0))  # 1.0 = d(x + y)/dy
    return new_node
end
function Base.:-(x::Node, y::Node)
    new_node = Node(x.value - y.value, nothing, [])
    push!(x.children, (new_node, 1.0))   # 1.0 = d(x - y)/dx
    push!(y.children, (new_node, -1.0))  # -1.0 = d(x - y)/dy
    return new_node
end
function Base.:*(x::Node, y::Node)
    new_node = Node(x.value * y.value, nothing, [])
    push!(x.children, (new_node, y.value))  # y.value = d(x * y)/dx
    push!(y.children, (new_node, x.value))  # x.value = d(x * y)/dy
    return new_node
end
function Base.sin(x::Node)
    new_node = Node(sin(x.value), nothing, [])
    push!(x.children, (new_node, cos(x.value)))  # cos(x.value) = d(sin(x))/dx
    return new_node
end
function Base.cos(x::Node)
    new_node = Node(cos(x.value), nothing, [])
    push!(x.children, (new_node, -sin(x.value)))  # -sin(x.value) = d(cos(x))/dx
    return new_node
end

function calc_derivative!(node::Node)
    if isnothing(node.deriv)
        node.deriv = sum([weight * calc_derivative!(child) for (child, weight) in node.children])
    end
    return node.deriv
end

function grad_reverse(f, inputs...)
    # Forward pass
    node_inputs = [Node(input, nothing, []) for input in inputs]
    output_node = f(node_inputs...)
    # Calculate the derivatives
    output_node.deriv = 1.0
    derivs = [calc_derivative!(node) for node in node_inputs]
    # Return value plus derivatives
    return output_node.value, derivs
end
