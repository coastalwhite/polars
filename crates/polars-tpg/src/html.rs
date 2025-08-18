pub static CDN_URL: &str =
    "https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.33.1/cytoscape.min.js";

use std::fmt;

use crate::{TextPlanGraph, TpgKey};

pub struct TpgHtml<'a> {
    pub plan: &'a TextPlanGraph,
}

impl<'a> fmt::Display for TpgHtml<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"
<html>
<head>
    <title>Plan</title>
    <style>
    body {{ 
      font: 14px helvetica neue, helvetica, arial, sans-serif;
    }}

    #cy {{
      height: 100%;
      width: 100%;
      position: absolute;
      left: 0;
      top: 0;
    }}
    </style>
    <script src="{CDN_URL}"></script>
    <script>
    function getTextWidth(text, font) {{
      const canvas = getTextWidth.canvas || (getTextWidth.canvas = document.createElement("canvas"));
      const context = canvas.getContext("2d");
      context.font = font;
      var width = 0;
      text.split("\n").forEach(function (item) {{
          const metrics = context.measureText(item);
          if (metrics.width > width) {{
              width = metrics.width;
          }}
        }});
      return width;
    }}

    window.addEventListener('DOMContentLoaded', function(){{
        var stylesheet = cytoscape.stylesheet();
                
        const fontFamily = "monospace";
        const fontSize = "20px";
        const nodes = [
        "#
        )?;
        let mut stack = Vec::new();
        stack.extend((0..self.plan.roots).map(|i| (<Option<TpgKey>>::None, TpgKey(i))));
        while let Some((p, n)) = stack.pop() {
            let node = &self.plan.nodes[n.0];
            write!(f, r#"{{ label: "{}", children: ["#, node.title.as_ref())?;
            for c in &node.children {
                write!(f, "{}, ", c.0)?;
            }
            f.write_str("]")?;
            if let Some(p) = p {
                write!(f, r#", parent: 'n{}'"#, p.0)?;
            }
            f.write_str(" },\n")?;

            stack.extend(node.children.iter().copied().map(|c| (p, c)));
        }
        f.write_str(r#"
        ];

        for (const [key, value] of nodes.entries()) {
            const label = value.label;
            const width = getTextWidth(label, `${fontSize} ${fontFamily}`);
            stylesheet = stylesheet.selector(`#n${key}`).css({
                'label': label,
                'width': `${width}px`,
            });
        }

        var cy = cytoscape({
            container: document.getElementById('cy'),
            elements: {
                nodes: nodes.map((_, idx) => ({ data: { id: `n${idx}` } })),
                edges: nodes.flatMap( (value, idx) => (value.children.map( (child) => ( { data: { source: `n${child}`, target: `n${idx}` } })))),
            },
            style: stylesheet
                .selector('node').css({
                    'font-family': fontFamily,
                    'font-size': fontSize,
                    'padding': '8px',
                    'shape': 'round-rectangle',
                    'border-color': '#000',
                    'border-width': 3,
                    'border-opacity': 0.5,
                    'background-color': '#eeeeee',
                    'text-color': '#ffffff',
                    'text-wrap': 'wrap',
                    'text-halign': 'center',
                    'text-valign': 'center'
                }),
            layout: {
                name: 'breadthfirst',
                directed: true,
                direction: 'upward',
            },
        });
    });
    </script>
</head>
<body><div id="cy" /></body>
</html>
        "#
        )?;

        Ok(())
    }
}
