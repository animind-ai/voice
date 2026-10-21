{{CONTEXT}}

这一段是整集的第 {{INDEX}} / {{COUNT}} 段，对应整集时间 {{WINDOW}}（片长 {{DURATION}}）。
JSON 里的 start 和 end 用秒，从这一段的第一帧算起。0 是这一段的开头，不是整集的开头。

只返回一个 JSON 对象，不要 markdown 代码块，不要额外解释。字段和取值必须符合这个 schema：

{{SCHEMA}}
