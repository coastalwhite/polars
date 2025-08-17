use std::fmt;

use crate::{TextPlanGraph, TpgKey};

pub struct TpgDot<'a> {
    pub plan: &'a TextPlanGraph,
}

impl<'a> fmt::Display for TpgDot<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut stack = Vec::new();

        stack.extend((0..self.plan.roots).map(|i| TpgKey(i)));

        f.write_str("digraph polars {\n  rankdir=\"BT\"\n  node [fontname=\"Monospace\"]\n")?;
        while let Some(n) = stack.pop() {
            let node = &self.plan.nodes[n.0];

            writeln!(f, "  {}[label=\"{}\"]", n.0, node.title)?;
            for c in node.children.iter().copied() {
                writeln!(f, "  {} -> {}", c.0, n.0)?;
            }
            stack.extend(node.children.iter().copied());
        }
        f.write_str("}")?;

        Ok(())
    }
}
