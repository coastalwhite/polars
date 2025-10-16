use super::*;

pub fn is_scalar_ae(node: Node, arena: &Arena<AExpr>) -> bool {
    arena.get(node).is_scalar(arena)
}

pub fn is_scalar_with_ctx_ae(node: Node, arena: &Arena<AExpr>, ctx: &ExprTraversalContext) -> bool {
    arena.get(node).is_scalar_with_ctx(arena, ctx)
}

pub fn is_length_preserving_ae(node: Node, arena: &Arena<AExpr>) -> bool {
    arena.get(node).is_length_preserving(arena)
}

pub fn is_length_preserving_with_ctx_ae(
    node: Node,
    arena: &Arena<AExpr>,
    ctx: &ExprTraversalContext,
) -> bool {
    arena.get(node).is_length_preserving_with_ctx(arena, ctx)
}

pub fn is_elementwise_ae(node: Node, arena: &Arena<AExpr>) -> bool {
    is_elementwise_with_ctx_ae(node, arena, &ExprTraversalContext::DEFAULT)
}

pub fn is_elementwise_with_ctx_ae(
    node: Node,
    arena: &Arena<AExpr>,
    ctx: &ExprTraversalContext,
) -> bool {
    arena.get(node).is_elementwise_with_ctx(arena, ctx)
}
