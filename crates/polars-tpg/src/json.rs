use std::fmt;

use crate::{TextPlanGraph, TpgKey, TpgValue};

pub struct TpgJson<'a> {
    pub plan: &'a TextPlanGraph,
}

impl<'a> fmt::Display for TpgJson<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut stack = Vec::new();

        stack.extend((0..self.plan.roots).map(|i| TpgKey(i)));

        f.write_str("{\n")?;
        writeln!(
            f,
            r#"  "title": "{}","#,
            WrapEscape(self.plan.title.as_ref())
        )?;
        writeln!(f, r#"  "roots": {},"#, self.plan.roots)?;
        writeln!(f, r#"  "nodes": ["#)?;
        while let Some(n) = stack.pop() {
            let node = &self.plan.nodes[n.0];

            write!(
                f,
                r#"    {{ "title": "{}", "arguments": ["#,
                WrapEscape(node.title.as_ref())
            )?;

            if let Some((_, fst)) = node.arguments.first() {
                write!(f, "{fst}")?;
                for (_, arg) in &node.arguments[1..] {
                    write!(f, ", {arg}")?;
                }
            }

            write!(f, r#"], "properties": {{"#)?;
            if let Some((_, key, value)) = node.properties.first() {
                write!(f, r#""{}": {value}"#, WrapEscape(key))?;
                for (_, key, value) in &node.properties[1..] {
                    write!(f, r#", "{}": {value}"#, WrapEscape(key))?;
                }
            }
            write!(f, r#"}}, "children": ["#)?;
            if let Some(k) = node.children.first() {
                write!(f, "{}", k.0)?;
                for k in &node.children[1..] {
                    write!(f, ", {}", k.0)?;
                }
            }
            f.write_str("]")?;
            if let Some(subnodes) = &node.subnodes {
                write!(
                    f,
                    r#", "subnodes": {{ "start": {}, "roots": {} }}"#,
                    subnodes.start.0, subnodes.roots
                )?;
            }
            f.write_str(" }")?;
            if !stack.is_empty() || !node.children.is_empty() {
                writeln!(f, ",")?;
            }

            stack.extend(node.children.iter().copied());
        }
        writeln!(f, r#"  ]"#)?;
        f.write_str("}")?;

        Ok(())
    }
}

impl fmt::Display for TpgValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TpgValue::Float(v) => v.fmt(f),
            TpgValue::Integer(v) => v.fmt(f),
            TpgValue::String(v) => write!(f, "\"{}\"", WrapEscape(v)),
            TpgValue::Binary(_) => todo!(),
            TpgValue::List(v) => {
                f.write_str("[")?;
                if let Some(fst) = v.first() {
                    write!(f, "{fst}")?;
                    for v in &v[1..] {
                        write!(f, ", {v}")?;
                    }
                }
                f.write_str("]")
            },
            TpgValue::Dictionary(items) => {
                f.write_str("{")?;
                if let Some((key, value)) = items.first() {
                    write!(f, r#""{}": {value}"#, WrapEscape(key))?;
                    for (key, value) in &items[1..] {
                        write!(f, r#", "{}": {value}"#, WrapEscape(key))?;
                    }
                }
                f.write_str("}")
            },
        }
    }
}

pub struct WrapEscape<'a>(&'a str);
impl<'a> fmt::Display for WrapEscape<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use fmt::Write;
        write!(EscapeLabel(f), "{}", self.0)
    }
}

/// Utility structure to write to a [`fmt::Formatter`] whilst escaping the output as a label name
pub struct EscapeLabel<'a>(pub &'a mut dyn fmt::Write);

impl fmt::Write for EscapeLabel<'_> {
    fn write_str(&mut self, mut s: &str) -> fmt::Result {
        loop {
            let mut char_indices = s.char_indices();

            // This escapes quotes and new lines
            // @NOTE: I am aware this does not work for \" and such. I am ignoring that fact as we
            // are not really using such strings.
            let f = char_indices.find_map(|(i, c)| match c {
                '"' => Some((i, r#"\""#)),
                '\n' => Some((i, r#"\n"#)),
                _ => None,
            });

            let Some((at, to_write)) = f else {
                break;
            };

            self.0.write_str(&s[..at])?;
            self.0.write_str(to_write)?;
            s = &s[at + 1..];
        }

        self.0.write_str(s)?;

        Ok(())
    }
}
