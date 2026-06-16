"""
ToolAgent: 给LLM装上工具 (读文件/搜代码/执行命令/抓网页)
=============================================================
DeepSeek: 主攻代码+文件操作   千问: 审查内容+检查结果
"""
import requests, json, time, os, sys, re, subprocess, glob as glob_mod
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ═══ 工具集 ═══
TOOLS = [
    {"name": "read_file", "desc": "读取文件内容",
     "params": {"path": "文件路径"}, "usage": 'read_file("C:/ai/README.md")'},
    {"name": "write_file", "desc": "写入文件",
     "params": {"path": "路径", "content": "内容"}, "usage": 'write_file("C:/ai/test.py", "print(1)")'},
    {"name": "search_files", "desc": "搜索文件(glob)",
     "params": {"pattern": "匹配模式", "path": "搜索目录"},
     "usage": 'search_files("**/*.py", "C:/ai")'},
    {"name": "search_content", "desc": "搜索文件内容(grep)",
     "params": {"pattern": "正则", "path": "搜索路径", "glob": "文件过滤"},
     "usage": 'search_content("class.*Model", "C:/ai", "*.py")'},
    {"name": "run_cmd", "desc": "执行命令",
     "params": {"cmd": "命令"}, "usage": 'run_cmd("dir C:/ai")'},
    {"name": "web_fetch", "desc": "抓取网页内容",
     "params": {"url": "URL"}, "usage": 'web_fetch("https://example.com")'},
    {"name": "list_dir", "desc": "列出目录",
     "params": {"path": "目录路径"}, "usage": 'list_dir("C:/ai")'},
]


# 参数别名映射: 容错LLM编错的参数名
PARAM_ALIASES = {
    "read_file": {"path": ["path", "filepath", "file_path", "fp"]},
    "write_file": {"path": ["path", "filepath", "file_path"], "content": ["content", "text", "data"]},
    "search_files": {"pattern": ["pattern", "glob"], "path": ["path", "base_path", "dir", "directory"]},
    "search_content": {"pattern": ["pattern", "regex", "query"], "path": ["path", "base_path", "dir"], "glob": ["glob", "file_filter", "filter"]},
    "run_cmd": {"cmd": ["cmd", "command", "exec", "shell"]},
    "web_fetch": {"url": ["url", "link", "address"]},
    "list_dir": {"path": ["path", "dir", "directory", "folder"]},
}

def resolve_params(tool_name, params):
    """将LLM可能编错的参数名映射到标准名 + 容错兜底"""
    aliases = PARAM_ALIASES.get(tool_name, {})
    resolved = {}
    # 1. 优先匹配标准名
    for std_name in aliases:
        if std_name in params:
            resolved[std_name] = params[std_name]
    # 2. 匹配别名
    for std_name, aliases_list in aliases.items():
        if std_name not in resolved:
            for alias in aliases_list:
                if alias in params:
                    resolved[std_name] = params[alias]
                    break
    # 3. 兜底: 智能推断 (处理LLM编造的参数名如 "file" 替代 "path")
    if not resolved:
        for k, v in params.items():
            if isinstance(v, str) and (':' in v or v.startswith('/') or '\\' in v):
                if tool_name in ("read_file", "write_file", "list_dir"):
                    resolved["path"] = v
                elif tool_name == "run_cmd":
                    resolved["cmd"] = v
                break
    return resolved


def execute_tool(name, params):
    """执行工具调用"""
    params = resolve_params(name, params)  # 容错参数名
    try:
        if name == "read_file":
            path = params.get("path", "")
            if not os.path.exists(path): return f"[错误] 文件不存在: {path}"
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return content  # 完整返回, 不截断

        elif name == "write_file":
            path = params.get("path", "")
            content = params.get("content", "")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"[OK] 已写入: {path} ({len(content)}字符)"

        elif name == "search_files":
            pattern = params.get("pattern", "*")
            search_path = params.get("path", ".")
            files = glob_mod.glob(os.path.join(search_path, pattern), recursive=True)
            return "\n".join(files[:50]) if files else "[无匹配]"

        elif name == "search_content":
            pattern = params.get("pattern", "")
            search_path = params.get("path", ".")
            file_filter = params.get("glob", "*.py")
            results = []
            for f in glob_mod.glob(os.path.join(search_path, "**", file_filter), recursive=True):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        for i, line in enumerate(fh, 1):
                            if re.search(pattern, line):
                                results.append(f"{f}:{i}: {line.strip()[:120]}")
                                if len(results) >= 30: break
                except: pass
                if len(results) >= 30: break
            return "\n".join(results) if results else "[无匹配]"

        elif name == "run_cmd":
            cmd = params.get("cmd", "echo ok")
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=30,
                                    encoding="utf-8", errors="replace")
            return (result.stdout + result.stderr)[:2000] or "[无输出]"

        elif name == "web_fetch":
            url = params.get("url", "")
            resp = requests.get(url, timeout=15, headers={"User-Agent": "ToolAgent/1.0"})
            text = resp.text[:5000]
            # 简单提取文本(去HTML标签)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text[:3000]

        elif name == "list_dir":
            path = params.get("path", ".")
            items = os.listdir(path)
            result = []
            for item in sorted(items)[:50]:
                full = os.path.join(path, item)
                tag = "[DIR]" if os.path.isdir(full) else f"[{os.path.getsize(full)}B]"
                result.append(f"  {tag} {item}")
            return "\n".join(result)

        return f"[未知工具] {name}"
    except Exception as e:
        return f"[工具异常] {name}: {str(e)}"


def build_tools_prompt():
    lines = [
        "【严格规则】你有工具可用。不要猜测 - 直接调用工具获取真实数据。",
        "工具调用必须用JSON: {\"tool\": \"工具名\", \"params\": {\"参数名\": \"值\"}}",
        "参数名必须用下面定义的精确名称, 不要自己编!",
        "",
        "可用工具:",
    ]
    for t in TOOLS:
        pnames = list(t["params"].keys())
        lines.append(f"  {t['name']}({', '.join(pnames)}): {t['desc']}")
        lines.append(f"    例: {t['usage']}")
    lines.extend([
        "",
        "【重要】read_file的参数是path不是filepath! run_cmd的参数是cmd不是exec!",
        "如果工具返回[错误], 检查参数名是否正确。",
    ])
    return "\n".join(lines)


class ToolAgent:
    """带工具+文件缓存的LLM Agent"""

    def __init__(self, name, api_key, base_url, model, role="assistant"):
        self.name = name; self.api_key = api_key; self.base_url = base_url.rstrip('/')
        self.model = model; self.role = role
        self.cache_file = f"C:/ai/logs/agent_cache_{name}.txt"
        self.memory_file = f"C:/ai/logs/agent_memory_{name}.txt"  # 永久记忆库
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        self.cache = []
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.cache = [line.strip() for line in f.readlines()[-30:]]
        self.messages = [{"role": "system", "content": build_tools_prompt()}]

    def _save_cache(self, key, content):
        entry = f"[{key}] {content[:200]}"  # 缓存摘要, 不影响LLM
        self.cache.append(entry)
        if len(self.cache) > 30: self.cache = self.cache[-30:]
        with open(self.cache_file, "w", encoding="utf-8") as f:
            f.write("\n".join(self.cache))

    def save_to_memory(self, entry):
        """保存到永久记忆库"""
        stamp = time.strftime("%m-%d %H:%M")
        with open(self.memory_file, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {entry[:2000]}\n\n")
        print(f"  [已存入记忆] {self.memory_file}")

    def load_memory(self):
        """加载记忆库最近内容"""
        if not os.path.exists(self.memory_file): return ""
        with open(self.memory_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "【永久记忆库】\n" + "".join(lines[-50:]) + "\n"

    def _get_cache_context(self):
        ctx = ""
        # 永久记忆
        mem = self.load_memory()
        if mem: ctx += mem
        # 临时缓存
        if self.cache:
            ctx += "【最近工具结果】\n" + "\n".join(self.cache[-10:]) + "\n"
        return ctx

    def chat(self, user_msg):
        """对话, 自动处理工具调用"""
        # 注入缓存上下文
        cache_ctx = self._get_cache_context()
        if cache_ctx:
            user_msg = cache_ctx + user_msg
        self.messages.append({"role": "user", "content": user_msg})
        return self._loop()

    def _loop(self, max_steps=None):
        """循环处理: LLM响应 → 检测工具调用 → 执行 → 反馈 → 循环"""
        if max_steps is None: max_steps = getattr(self, 'max_steps', 50)
        steps = 0; called = set()
        while steps < max_steps:
            steps += 1
            resp = self._call_api()
            if not resp: return "[错误] API无响应"

            tool_calls = self._extract_tools(resp)
            if not tool_calls:
                self.messages.append({"role": "assistant", "content": resp})
                return resp

            # 去重: 同一工具同一参数不重复调
            results = []
            for tc in tool_calls:
                tname = tc.get("tool", ""); tparams = tc.get("params", {})
                key = f"{tname}:{json.dumps(tparams, sort_keys=True)}"
                if key in called:
                    print(f"  [{self.name}] [SKIP] 跳过重复: {tname}")
                    continue
                called.add(key)
                print(f"  [{self.name}] [TOOL] {tname}({tparams})")
                r = execute_tool(tname, tparams)
                # 不截断, 完整传给LLM
                self._save_cache(tname, r[:200])
                results.append(f"[{tname}结果]\n{r}")

            if not results:
                self.messages.append({"role": "assistant", "content": resp})
                self.messages.append({"role": "user", "content": "工具已执行过, 请基于之前的结果直接回答, 不要再调用同样工具。"})
                continue

            feedback = "\n\n---\n".join(results)
            self.messages.append({"role": "assistant", "content": resp})  # 不截断
            self.messages.append({"role": "user", "content": feedback})

        return f"[超时] {steps}步内未完成"

    def _call_api(self):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"model": self.model, "messages": self.messages[-40:], "max_tokens": 16384, "temperature": 0.7}
        try:
            print(f"  [...] 等待{self.name}...", end="", flush=True)
            timeout = getattr(self, 'api_timeout', 600)
            r = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data, timeout=timeout)
            print(f"\r", end="", flush=True)  # 清除等待提示
            if r.status_code == 200:
                body = r.json()
                content = body["choices"][0]["message"]["content"]
                if body["choices"][0].get("finish_reason") == "length":
                    content += "\n\n[截断警告: 输出超max_tokens限制, 内容不完整。请精简或分段。]"
                return content
            return f"[API错误 {r.status_code}]"
        except Exception as e:
            return f"[网络错误: {e}]"

    def _extract_tools(self, text):
        """从回复中提取工具调用 (JSON + XML格式)"""
        calls = []
        # JSON格式: {"tool": "...", "params": {...}}
        for m in re.finditer(r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"params"\s*:\s*\{[^{}]*\}[^{}]*\}', text):
            try: calls.append(json.loads(m.group()))
            except: pass
        # JSON数组: {"tools": [...]}
        m = re.search(r'"tools"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if m:
            try:
                arr = json.loads(f"{{{m.group()}}}")["tools"]
                calls.extend(arr)
            except: pass
        # 千问XML格式: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
        for m in re.finditer(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
            block = m.group(1)
            func_m = re.search(r'<function=(\w+)>', block)
            if func_m:
                tname = func_m.group(1)
                params = {}
                for pm in re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', block, re.DOTALL):
                    params[pm.group(1)] = pm.group(2).strip()
                calls.append({"tool": tname, "params": params})
        return calls


def debate(ds, qw, question, max_rounds=3):
    """双Agent辩论: 用户打断时while重启(非递归)"""
    while True:
        print(f"\n{'='*50}\n[DeepSeek 第1轮] 分析中...\n{'='*50}")
        ds_resp = ds.chat(question)
        print(ds_resp)
        interrupted = False

        for r in range(1, max_rounds + 1):
            # 用户打断窗口
            print(f"\n  (按Enter打断, 3秒...)", end="", flush=True)
            try:
                import msvcrt
                for _ in range(30):
                    if msvcrt.kbhit() and msvcrt.getch() == b'\r':
                        inject = input("\n  [打断] 输入新想法: ").strip()
                        if inject:
                            question += f"\n[用户补充]: {inject}"
                            print(f"  [已注入] 重新开始...")
                            interrupted = True
                        break
                    time.sleep(0.1)
            except: pass
            if interrupted: break

            # 千问审查 (可调工具读代码验证)
            print(f"\n{'─'*40}\n[千问 第{r}轮审查] (可调工具验证, 稍等...)\n{'─'*40}")
            qw_resp = qw.chat(
                f"用户问题: {question}\n\nDeepSeek分析:\n{ds_resp}\n\n"
                f"请审查: 1)事实错误 2)遗漏 3)可优化处。可以调工具读代码验证, 给出3-5条建议。"
            )
            # 千问失败重试一次
            if not qw_resp or "[错误]" in str(qw_resp) or "[超时]" in str(qw_resp):
                print(f"  [千问首次无响应, 重试...]")
                qw_resp = qw.chat(
                    f"用户问题: {question}\n\nDeepSeek分析:\n{ds_resp}\n\n"
                    f"请审查: 1)事实错误 2)遗漏 3)可优化处。可调工具读代码验证。"
                )
            print(qw_resp)

            # 收敛判断
            if r >= 2:
                check = qw.chat(f"方案:\n{ds_resp}\n\n已完善? 是回CONVERGE, 否简述。")
                if "CONVERGE" in check.upper():
                    print(f"\n[收敛] 千问认可, 结束")
                    break

            # DeepSeek改进
            print(f"\n{'─'*40}\n[DeepSeek 第{r+1}轮] 吸收审查意见...\n{'─'*40}")
            refine = f"千问审查意见:\n{qw_resp}\n\n请基于以上意见改进你的分析。有道理的吸收, 不同意的说明理由。不需要工具时直接回答。"
            ds_resp = ds.chat(refine)
            print(ds_resp)

        if interrupted:
            continue  # 回到while起点重新开始

        # 正常结束
        try:
            save = input("\n  [存入记忆? y/n]: ").strip().lower()
            if save == 'y':
                ds.save_to_memory(f"Q: {question[:200]}\nA: {ds_resp[:1500]}")
                qw.save_to_memory(f"审查: Q={question[:200]}")
        except: pass
        return ds_resp


# ═══ 主程序 ═══
if __name__ == "__main__":
    ds = ToolAgent("DeepSeek",
        api_key=os.getenv("DS_API_KEY", "sk-1daad371c12e4c0d97fc5c44fe0f34c6"),
        base_url="https://api.deepseek.com/v1", model="deepseek-chat")
    qw = ToolAgent("千问",
        api_key=os.getenv("QW_API_KEY", "sk-3940b50f25a945789e6438ad11e434f8"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen3.7-plus")
    qw.max_steps = 40    # 千问读文件多, 步数多给
    qw.api_timeout = 1200  # 千问慢, 超时拉长

    print("=" * 50)
    print("  ToolAgent v3: 自动辩论模式")
    print("  输入→DeepSeek分析→千问审查→自动循环")
    print("  按Enter打断 | exit退出 | ds/千问 单独对话")
    print("=" * 50)

    while True:
        try:
            user = input(f"\n[>] ").strip()
            if user.lower() == "exit": break
            if not user: continue

            # 单独对话模式
            if user.lower() == "ds":
                q = input("[DeepSeek]> ").strip()
                if q: print(f"\n{ds.chat(q)}")
                continue
            if user.lower() == "千问" or user.lower() == "qw":
                q = input("[千问]> ").strip()
                if q: print(f"\n{qw.chat(q)}")
                continue

            # 默认: 自动辩论
            debate(ds, qw, user)

        except KeyboardInterrupt:
            print("\n[退出]")
            break
