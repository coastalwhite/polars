use polars::lazy::dsl::Operator;
use polars::prelude::*;

use crate::{Context, EntropySrc};
use rand::Rng;

bitset! {
    pub struct ExprSet, pub enum ExprVariant: u64 {
        Add,
        Sub,
        Mul,
        Div,
    }
}

fn arbitrary_expr_leaf(
    entropy: &mut EntropySrc,
    _context: &Context,
    _enabled: ExprSet,
) -> Result<Expr, ()> {
    // let expr = enabled.select_modulo(entropy.gen()).ok_or(())?;

    Ok(Expr::Literal(LiteralValue::Int32(entropy.gen())))
}

fn arbitrary_expr(
    entropy: &mut EntropySrc,
    context: &Context,
    enabled: ExprSet,
) -> Result<Expr, ()> {
    if context.expr.max_size == 0 {
        return Err(());
    }

    let mut current = arbitrary_expr_leaf(entropy, context, enabled)?;

    let mut length = 1;

    while entropy.gen::<u32>() % context.expr.max_size > length {
        let Some(variant) = enabled.select_modulo(entropy.gen()) else {
            return Ok(current);
        };

        let lit = arbitrary_expr_leaf(entropy, context, enabled)?;

        let (left, right) = if entropy.gen() {
            (current, lit)
        } else {
            (lit, current)
        };
        
        let left = Arc::new(left);
        let right = Arc::new(right);

        current = match variant {
            ExprVariant::Add => Expr::BinaryExpr { left, op: Operator::Plus, right },
            ExprVariant::Sub => Expr::BinaryExpr { left, op: Operator::Minus, right },
            ExprVariant::Mul => Expr::BinaryExpr { left, op: Operator::Multiply, right },
            ExprVariant::Div => Expr::BinaryExpr { left, op: Operator::Divide, right },
        };

        length += 1;
    }

    Ok(current)
}
