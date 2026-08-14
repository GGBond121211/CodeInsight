from __future__ import annotations

# The case catalog intentionally keeps each long natural-language question on one source line.
# ruff: noqa: E501
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = Path(__file__).resolve().parent
SOURCE = EVAL_ROOT / "multilingual_cases.json"
TARGET = EVAL_ROOT / "master_200_cases.json"


REPOSITORIES: dict[str, dict[str, Any]] = {
    "sample_repo": {
        "local_root": "backend/tests/fixtures/sample_repo",
        "upstream_url": None,
        "version": "task11-v2-fixture",
        "commit": None,
        "license": "project-internal test fixture",
    },
    "httpx": {
        "local_root": "work/benchmarks/httpx/httpx",
        "upstream_url": "https://github.com/encode/httpx",
        "version": "0.28.1",
        "commit": "26d48e0634e6ee9cdc0533996db289ce4b430177",
        "license": "BSD-3-Clause",
    },
    "click": {
        "local_root": "work/benchmarks/click/src/click",
        "upstream_url": "https://github.com/pallets/click",
        "version": "8.4.1",
        "commit": "6eeb50e948ea136db145280f6f5dd52eca3fa7e5",
        "license": "BSD-3-Clause",
    },
    "requests": {
        "local_root": "work/benchmarks/requests/src/requests",
        "upstream_url": "https://github.com/psf/requests",
        "version": "2.34.2",
        "commit": "6e83187b8feb273ed4c6cdab5efd8d54901dfab3",
        "license": "Apache-2.0",
    },
}


def evidence(path: str, start: int, end: int) -> dict[str, Any]:
    return {"path": path, "start_line": start, "end_line": end}


def chunk_compatible_segments(item: dict[str, Any], max_lines: int = 80) -> list[dict[str, Any]]:
    """Split one evidence requirement at the product's fixed chunk boundaries."""
    segments = []
    start = item["start_line"]
    while start <= item["end_line"]:
        chunk_end = ((start - 1) // max_lines + 1) * max_lines
        end = min(item["end_line"], chunk_end)
        segments.append(evidence(item["path"], start, end))
        start = end + 1
    return segments


def scenario(
    slug: str,
    category: str,
    protects: str,
    evidence_items: list[dict[str, Any]],
    required_terms: list[str],
    zh: str,
    mixed: str,
    features: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "slug": slug,
        "category": category,
        "protects": protects,
        "evidence": evidence_items,
        "required_terms": required_terms,
        "features": features
        or [
            "colloquial_fragment",
            "self_correction",
            "mixed_intent",
            "cross_file_or_branch_reasoning",
            "terminology_noise",
        ],
        "questions": [("zh", zh), ("zh-en", mixed)],
    }


HTTPX = [
    scenario(
        "top-level-get-dispatch",
        "cross_file_trace",
        "区分便捷 API、通用 request 和 Client 生命周期。",
        [evidence("_api.py", 39, 120), evidence("_api.py", 174, 207)],
        ["request", "Client", "GET"],
        "我看到 `get` 里面没真正发 socket，所以它是不是只拼 URL？别只报定义位置：从顶层 get 到通用 request，再到临时 Client 怎么进入发送流程，响应是谁返回的？",
        "`httpx.get()` looks too thin，我一开始以为是 alias，后来又看到 request；请把 top-level GET → request → Client 的实际 dispatch 和资源关闭边界串起来。",
    ),
    scenario(
        "request-client-lifecycle",
        "data_flow",
        "解释顶层 request 的临时客户端配置与上下文生命周期。",
        [evidence("_api.py", 39, 120)],
        ["Client", "with", "request"],
        "线上偶发连接没复用，是不是顶层 `request()` 偷偷用了全局连接池？请按源码说明它怎样构造 Client、哪些配置被传进去、with 结束后发生什么。",
        "调用 `httpx.request` 时 connection pool 是 singleton 吗？我怀疑搞反了；trace 一下 temporary Client context、send 和 close 的 ownership。",
    ),
    scenario(
        "build-request-merge",
        "data_flow",
        "覆盖 URL、headers、cookies、params 与 timeout 的请求级/客户端级合并。",
        [
            evidence("_client.py", 340, 389),
            evidence("_client.py", 391, 455),
            evidence("_client.py", 584, 591),
        ],
        ["build_request", "merge", "timeout"],
        "Client 上已有 base_url、header、cookie 和 timeout，我这次又传一份；不是问谁覆盖谁这么简单，`build_request` 具体按什么顺序合并、timeout 何时塞进 extensions？",
        "request-level params 跟 client defaults 打架时到底 merge 还是 replace？把 `build_request`、URL/header/cookie merge 和 timeout extension 的 chain 一起解释。",
    ),
    scenario(
        "relative-base-url",
        "boundary_behavior",
        "验证相对 URL 如何与 base_url 组合而非字符串拼接。",
        [evidence("_client.py", 391, 411), evidence("_urls.py", 354, 366)],
        ["base_url", "join", "URL"],
        "base_url 末尾有路径、请求又传 `../users`，我同事说直接字符串相加；请纠正这个前提，实际由谁判断相对地址并 join，绝对 URL 又怎么处理？",
        "`base_url + url` 是 plain concat 吗？例如 `../v2` 带 dot segment 时，show the `_merge_url` → `URL.join` behavior and absolute-URL bypass。",
    ),
    scenario(
        "auth-precedence",
        "boundary_behavior",
        "覆盖请求显式 auth、客户端 auth、URL 凭据与空认证优先级。",
        [evidence("_client.py", 412, 473)],
        ["auth", "URL", "USE_CLIENT_DEFAULT"],
        "这次请求传 `auth=None`，但 Client 和 URL 里都有凭据，我已经绕晕了：到底谁赢，None 是禁用还是回退？请从 auth 参数解析到 `_build_request_auth` 说清楚。",
        "client auth、URL userinfo、per-request auth 三套同时出现；`None` / `USE_CLIENT_DEFAULT` semantics 别混在一起，给出 precedence 和最终 Auth 对象来源。",
    ),
    scenario(
        "redirect-method-rewrite",
        "boundary_behavior",
        "验证 301/302/303/307/308 下方法重写差异。",
        [evidence("_client.py", 494, 515)],
        ["303", "GET", "HEAD"],
        "POST 被 302 后为什么变 GET，但 307 又没变？别泛讲 HTTP 标准，按 `_redirect_method` 的分支列出 301、302、303 的特殊处理和保留条件。",
        "redirect 后 method rewrite 看着 inconsistent：POST+302、任意 method+303、POST+301、307/308 分别怎样？以 implementation branches 回答。",
    ),
    scenario(
        "redirect-header-stripping",
        "boundary_behavior",
        "覆盖跨源认证头剥离及 GET 重写后的实体头删除。",
        [evidence("_client.py", 516, 571)],
        ["Authorization", "Content-Length", "Cookie"],
        "跳转后 Authorization、Cookie、Content-Length 是不是一刀切全保留？请区分跨源、方法改成 GET、Host 和 cookie 重建这几种情况。",
        "On redirect，哪些 headers copy、哪些 strip？尤其 cross-origin Authorization 和 body headers after POST→GET，不要把 Cookie handling 漏掉。",
    ),
    scenario(
        "redirect-loop",
        "cross_file_trace",
        "追踪同步发送、认证流与多次重定向的控制权。",
        [evidence("_client.py", 879, 999)],
        ["auth_flow", "redirect", "history"],
        "`send` 是不是自己 while 重定向直到结束？我看到 auth 也会 yield 多个 request；请把 send→认证→redirect 两层生成器关系、history 怎么累积讲清。",
        "sync Client.send has auth_flow and redirect flow，两个 loop 谁包谁？需要解释 response history、follow_redirects=False 时的 next_request。",
    ),
    scenario(
        "status-error-classification",
        "boundary_behavior",
        "解释请求缺失、状态类别与异常消息构造。",
        [evidence("_models.py", 771, 829)],
        ["HTTPStatusError", "request", "status"],
        "Response 只要不是 2xx 就一定抛吗？还有手工 new 的 Response 没 request 会怎样；按 `raise_for_status` 的先后检查和 1xx/3xx/4xx/5xx 分类回答。",
        "`raise_for_status()` only errors on 4xx/5xx？这个 assumption 可能错了；说明 missing request guard、status classes 和 HTTPStatusError context。",
    ),
    scenario(
        "sync-auth-feedback",
        "cross_file_trace",
        "验证同步认证流如何接收响应并可重发请求。",
        [evidence("_auth.py", 38, 85), evidence("_client.py", 930, 962)],
        ["sync_auth_flow", "send", "response"],
        "自定义 Auth 第一次 yield 请求后，服务器回 401，response 怎么送回认证生成器让它再 yield？同步链路里谁读 body、谁真正 dispatch？",
        "custom auth is a generator，不是 callback 一次结束：trace `sync_auth_flow`, response feedback, optional body read, and `_send_handling_redirects`。",
    ),
    scenario(
        "async-auth-feedback",
        "cross_file_trace",
        "验证异步认证流与异步发送路径，防止误引同步实现。",
        [evidence("_auth.py", 87, 110), evidence("_client.py", 1605, 1636)],
        ["async_auth_flow", "await", "response"],
        "AsyncClient 遇到 challenge 会不会偷跑同步 auth_flow？请只沿异步路径说明 request/response 如何在 async generator 之间来回，何时 await 读取响应。",
        "For AsyncClient auth retry，do not cite sync send：show async_auth_flow feedback loop and where response body is awaited before next request。",
    ),
    scenario(
        "sync-transport-dispatch",
        "cross_file_trace",
        "追踪同步 Client 从单请求发送到 transport.handle_request。",
        [evidence("_client.py", 1001, 1034)],
        ["handle_request", "BaseTransport", "Response"],
        "我只想确认真正越过网络边界的位置：同步 Client 在 hooks 之后怎样把 Request 交给 transport、再把底层响应包装并绑定 request？",
        "Where is the actual sync I/O boundary—not `Client.send` broadly? Trace `_send_single_request` → `transport.handle_request` → Response wrapping and cookie extraction。",
    ),
    scenario(
        "async-transport-dispatch",
        "cross_file_trace",
        "追踪 AsyncClient 到 handle_async_request，避免同步/异步混淆。",
        [evidence("_client.py", 1717, 1749)],
        ["handle_async_request", "AsyncByteStream", "await"],
        "异步发送最终调用的是线程池里的同步 transport 吗？按源码指出请求流类型检查、await 的 transport 方法和返回流包装。",
        "Async transport path 到底是 `handle_request` 还是 `handle_async_request`？include AsyncByteStream assertions, await boundary, response construction。",
    ),
    scenario(
        "decoded-line-stream",
        "data_flow",
        "解释字节解压、文本解码与跨 chunk 行切分状态。",
        [evidence("_models.py", 884, 933), evidence("_decoders.py", 321, 378)],
        ["iter_bytes", "iter_text", "LineDecoder"],
        "服务端一行被拆到两个 chunk，甚至 `\r` 和 `\n` 分开，`iter_lines` 会不会吐半行？把 bytes 解码、text chunk、LineDecoder 缓冲和 flush 顺序说全。",
        "stream chunks ≠ logical lines；trace `iter_bytes → iter_text → LineDecoder`, including split CRLF and trailing buffered text。",
    ),
    scenario(
        "decoder-stack-order",
        "data_flow",
        "验证多 Content-Encoding 解码器的逆序应用。",
        [evidence("_decoders.py", 203, 225), evidence("_decoders.py", 381, 393)],
        ["MultiDecoder", "reversed", "Content-Encoding"],
        "响应头写了多层 Content-Encoding，代码是按 header 顺序还是反过来解？为什么 flush 也不能随便正序；请从 decoder 组装到 MultiDecoder 说明。",
        "For `gzip, deflate` style stacked encodings，decode order 容易讲反：show decoder creation and why `MultiDecoder` stores them reversed。",
    ),
]


CLICK = [
    scenario(
        "context-default-map",
        "boundary_behavior",
        "解释 default_map callable 的求值开关。",
        [evidence("core.py", 742, 771)],
        ["default_map", "callable", "lookup_default"],
        "default_map 放了函数，是注册时就执行还是取值时执行？`call=False` 又返回啥；别把参数默认值和 Context 默认表混为一谈。",
        "`Context.lookup_default` 遇到 callable default：when is it invoked, and what does `call=False` preserve?",
    ),
    scenario(
        "context-invoke-forward",
        "semantic_paraphrase",
        "区分 invoke 显式参数与 forward 继承当前参数。",
        [evidence("core.py", 823, 896)],
        ["invoke", "forward", "params"],
        "两个命令互调时我把 `ctx.invoke` 和 `ctx.forward` 当同义词了；哪个只吃显式 kwargs，哪个会把当前 ctx.params 补进去，冲突时谁覆盖？",
        "Click command delegation: `invoke` vs `forward` are not aliases—explain inherited params, explicit overrides, and context creation。",
    ),
    scenario(
        "make-context",
        "data_flow",
        "覆盖上下文创建、额外参数配置与解析失败时关闭。",
        [evidence("core.py", 1221, 1256)],
        ["make_context", "parse_args", "close"],
        "`make_context` 只是 new Context 吗？如果 parse_args 中途抛错，资源会不会漏；还有 resilient parsing 是在哪个阶段生效的？",
        "Trace `Command.make_context`: extra settings, context scope, parsing, and cleanup when parse raises。",
    ),
    scenario(
        "command-main-exits",
        "boundary_behavior",
        "覆盖 standalone_mode 下异常、退出码和返回值语义。",
        [evidence("core.py", 1377, 1488)],
        ["standalone_mode", "SystemExit", "ClickException"],
        "CLI 回调返回 7，`main()` 就以 7 退出吗？请区分 standalone_mode 开关、ClickException、Abort、EOF/KeyboardInterrupt 和正常返回。",
        "`Command.main` return value vs process exit got mixed up；explain standalone_mode, SystemExit conversion, ClickException and abort handling。",
    ),
    scenario(
        "group-chain-invoke",
        "cross_file_trace",
        "解释链式 Group 如何逐个建子上下文并聚合结果。",
        [evidence("core.py", 1864, 1944)],
        ["chain", "sub_ctx", "result_callback"],
        "chain=True 时是先解析完所有子命令再一起跑，还是边解析边执行？group callback、每个 subcommand 和 result_callback 的先后顺序是什么？",
        "For a chained Group，trace parse contexts → group callback → each command invoke → result_callback; also say what result shape is passed。",
    ),
    scenario(
        "group-resolve-command",
        "boundary_behavior",
        "覆盖命令 token 正规化、未知命令和无参数恢复。",
        [evidence("core.py", 1946, 1970)],
        ["resolve_command", "token_normalize_func", "UsageError"],
        "用户输的子命令大小写不对，Click 是先报 unknown 还是先走 token_normalize_func？若根本没参数，返回结构又是什么？",
        "`resolve_command` with a misspelled/case-normalized token: lookup order, normalized retry, UsageError, and resilient parsing behavior。",
    ),
    scenario(
        "parameter-source-precedence",
        "data_flow",
        "覆盖命令行、环境变量、default_map 与默认值优先级。",
        [evidence("core.py", 2338, 2384)],
        ["COMMANDLINE", "ENVIRONMENT", "DEFAULT_MAP"],
        "同一个 option 在命令行、envvar、default_map 和 default 都有值，我只知道命令行大概最高；请按 consume_value 的实际判断顺序和 ParameterSource 标记回答。",
        "CLI/env/default_map/default precedence and provenance：show exact `consume_value` order and when source stays DEFAULT。",
    ),
    scenario(
        "parameter-type-shape",
        "boundary_behavior",
        "解释 multiple、nargs=-1、组合类型与缺失值的形状。",
        [evidence("core.py", 2386, 2440)],
        ["multiple", "nargs", "BadParameter"],
        "一个参数同时 multiple、nargs>1 时返回几层 tuple？空值是不是永远 None？请把 type_cast_value 对 nargs=-1、固定 nargs 和长度不匹配的行为拆开。",
        "Parameter shape puzzle: `multiple`, fixed `nargs`, `nargs=-1`, composite type, missing input—explain conversion nesting and BadParameter boundary。",
    ),
    scenario(
        "parameter-process",
        "data_flow",
        "追踪值转换、required 检查与 callback 的执行顺序。",
        [evidence("core.py", 2460, 2524), evidence("core.py", 2587, 2673)],
        ["process_value", "required", "callback"],
        "required 检查是在 type conversion 前还是后？callback 抛错又在哪层；从 parser 结果到 ctx.params 落值，按真实顺序说。",
        "Trace parse result → consume → type cast → missing/required check → callback → ctx.params; avoid claiming callback runs on raw tokens。",
    ),
    scenario(
        "option-flag-prompt",
        "boundary_behavior",
        "覆盖 flag_value、多次出现、prompt 与默认值交互。",
        [evidence("core.py", 3359, 3436)],
        ["flag_value", "prompt", "DEFAULT"],
        "一个 option 既是 flag 又配置 prompt，多次出现还可能带 value；什么时候直接取 flag_value，什么时候真的询问，默认值会不会误触发 prompt？",
        "Flag + prompt + repeated option is messy：explain `Option.consume_value` special cases and later `process_value` prompt decision。",
    ),
    scenario(
        "decorator-context-object",
        "cross_file_trace",
        "区分 pass_context、pass_obj 与类型对象查找。",
        [evidence("decorators.py", 28, 97)],
        ["pass_context", "pass_obj", "find_object"],
        "装饰器到底把 Context 本身、ctx.obj 还是父级里某种对象塞给函数？请比较 pass_context、pass_obj、make_pass_decorator(ensure=True)。",
        "`pass_context`, `pass_obj`, typed `make_pass_decorator` 三者 injection source 不同；include parent lookup and ensure behavior。",
    ),
    scenario(
        "choice-normalization",
        "boundary_behavior",
        "解释 Choice 的正规化、大小写与原始值返回。",
        [evidence("types.py", 332, 390)],
        ["normalize_choice", "case_sensitive", "choices"],
        "Choice(case_sensitive=False) 匹配到大小写不同的输入后，是返回用户字符串还是 choices 里的原对象？normalize 又在哪一步做？",
        "Case-insensitive `Choice` should not necessarily return normalized input；trace mapping construction, token normalization and returned original choice。",
    ),
    scenario(
        "choice-completion",
        "semantic_paraphrase",
        "验证 shell completion 使用正规化前缀但返回原 choice。",
        [evidence("types.py", 409, 429)],
        ["shell_complete", "CompletionItem", "incomplete"],
        "补全输入 `fo` 时如果 choice 是 `Foo`，大小写策略和 token_normalize_func 谁参与？展示值为什么仍保持原样。",
        "Shell completion prefix matching for Choice: normalized incomplete vs normalized choice, while CompletionItem keeps original value。",
    ),
    scenario(
        "file-lazy-atomic",
        "boundary_behavior",
        "覆盖 File 类型的懒打开、原子写入、标准流与错误处理。",
        [evidence("types.py", 863, 908)],
        ["LazyFile", "open_stream", "atomic"],
        "File(lazy=True, atomic=True) 是 convert 时就建临时文件吗？`-` 标准流和已经打开的 file object 又走什么分支，错误会被谁包装？",
        "Click `File.convert` path matrix: existing file object, `-`, lazy, atomic, and open error—what gets registered for cleanup?",
    ),
    scenario(
        "path-validation",
        "boundary_behavior",
        "解释 Path 的 exists、file/dir、权限与 resolve_path 顺序。",
        [evidence("types.py", 1028, 1099)],
        ["resolve_path", "exists", "BadParameter"],
        "路径不存在但 exists=False 时，是直接放行还是还做 readable/writable 检查？resolve_path、stat、file_okay/dir_okay 的顺序请按代码说明。",
        "`Path.convert` with missing path and `exists=False`: resolve, stat failure, type checks, permissions and value_type conversion order。",
    ),
    scenario(
        "runner-isolation",
        "cross_file_trace",
        "覆盖 CliRunner 对输入输出、环境变量、颜色和全局状态的隔离。",
        [evidence("testing.py", 363, 558)],
        ["isolation", "stdin", "environ"],
        "CliRunner.isolation 只是 redirect stdout 吗？它会改 env、stdin、stderr、颜色和 terminal width；退出后哪些全局对象必须恢复？",
        "Testing isolation mutates process-global state；trace stdin/stdout/stderr wrappers, env patching, color/width, and finally restoration。",
    ),
    scenario(
        "runner-invoke",
        "data_flow",
        "解释 invoke 参数解析、异常捕获和 Result 组装。",
        [evidence("testing.py", 560, 703)],
        ["catch_exceptions", "Result", "SystemExit"],
        "测试里命令抛 SystemExit(0)、Click 异常或普通异常，invoke 都算 exception 吗？output、return_value、exit_code 怎么组合进 Result？",
        "`CliRunner.invoke` outcome matrix: normal return, SystemExit, arbitrary exception with catch_exceptions on/off, and captured streams。",
    ),
    scenario(
        "shell-completion",
        "cross_file_trace",
        "追踪 Command shell_complete 对参数和子命令建议的组合。",
        [evidence("core.py", 1310, 1355), evidence("shell_completion.py", 519, 566)],
        ["shell_complete", "incomplete", "CompletionItem"],
        "补全时前面的 token 已经解析了一半，Click 怎么决定让当前参数补还是让命令本身补？不要只说调用 shell_complete，要说明 incomplete 的归属。",
        "Completion dispatch with incomplete args: resolve context, identify object, then collect parameter/command CompletionItems—trace both sides。",
    ),
]


REQUESTS = [
    scenario(
        "top-level-api-session",
        "cross_file_trace",
        "区分顶层 API 与持久 Session 的生命周期。",
        [evidence("api.py", 24, 71)],
        ["Session", "with", "request"],
        "`requests.get` 会不会复用一个全局 Session？我看到连接池又像能复用；从 get/request 到临时 Session 的创建、发送和关闭说清楚。",
        "Top-level `requests.request()` pooling assumption：trace temporary Session context and explain why it differs from reusing your own Session。",
    ),
    scenario(
        "prepare-request-merge",
        "data_flow",
        "覆盖 Session 与 Request 的 cookies、headers、auth、hooks 合并。",
        [evidence("sessions.py", 511, 555)],
        ["PreparedRequest", "merge_setting", "merge_hooks"],
        "Session 里有 cookie/header/auth/hooks，本次 Request 又传一套；prepare_request 到底先准备 CookieJar 还是先合并设置，哪些字段不是简单覆盖？",
        "Trace `Session.prepare_request`: cookie jar merge, auth fallback/netrc, headers, params and hooks into PreparedRequest。",
    ),
    scenario(
        "session-request-pipeline",
        "cross_file_trace",
        "追踪 Session.request 从 Request 到 PreparedRequest、环境设置和 send。",
        [evidence("sessions.py", 557, 653), evidence("sessions.py", 831, 868)],
        ["prepare_request", "merge_environment_settings", "send"],
        "Session.request 里 timeout、verify、proxy 是在 prepare 前写入 Request 吗？请按 Request→prepare→环境合并→send 的真实数据流回答。",
        "`Session.request` pipeline: construct Request, prepare it, merge environment settings, then send; place timeout/allow_redirects correctly。",
    ),
    scenario(
        "session-send",
        "cross_file_trace",
        "覆盖 adapter 选择、hooks、cookie、重定向与流内容消费。",
        [evidence("sessions.py", 752, 829)],
        ["get_adapter", "dispatch_hook", "resolve_redirects"],
        "真正 send 后，是先抽 cookie 还是先跑 response hook？redirect history 何时塞回 response，stream=False 又在哪里强制读 body？",
        "Order inside `Session.send`: adapter, elapsed, response hooks, cookies, redirects/history, and eager content when not streaming。",
    ),
    scenario(
        "merge-setting-none",
        "boundary_behavior",
        "解释映射合并时 None 删除键及非映射设置的选择。",
        [evidence("sessions.py", 76, 106)],
        ["merge_setting", "None", "Mapping"],
        "请求 headers 里某键设 None，是覆盖成 None 还是从 Session 默认里删除？如果一边根本不是 Mapping，规则又变成什么？",
        "`merge_setting` is not plain dict.update：explain None inputs, non-Mapping values, OrderedDict merge and null-key removal。",
    ),
    scenario(
        "merge-hooks-empty",
        "boundary_behavior",
        "验证空 response hooks 不会意外清空另一侧 hooks。",
        [evidence("sessions.py", 108, 124)],
        ["response", "hooks", "merge_setting"],
        "我传 `hooks={'response': []}` 是想禁用 Session hook，但源码好像不让；空列表为什么会回退，哪一侧为空分别怎样？",
        "Empty response-hook list has special semantics：request empty vs session empty, then fallback to merge_setting。",
    ),
    scenario(
        "strip-auth",
        "boundary_behavior",
        "覆盖跨主机、默认端口与 http→https 重定向认证剥离。",
        [evidence("sessions.py", 154, 184)],
        ["hostname", "default_port", "Authorization"],
        "重定向从 http://a 到 https://a 会不会删 Authorization？端口显式 80/443 又怎样；请区分同主机升级特例、跨主机和端口变化。",
        "`should_strip_auth` matrix: host change, http→https default ports exception, scheme/port changes, implicit vs explicit ports。",
    ),
    scenario(
        "redirect-resolution",
        "cross_file_trace",
        "解释重定向循环中的 URL、method、cookie、body 与代理重建。",
        [evidence("sessions.py", 186, 307)],
        ["Location", "rebuild_method", "rewind_body"],
        "302 链里下一跳不是简单改 URL：fragment、relative Location、method、body headers、cookie、proxy 和 rewind 分别在哪一步处理？",
        "Trace one `resolve_redirects` iteration from Location normalization through method/body/cookies/proxies/auth to optional rewind and send。",
    ),
    scenario(
        "redirect-method",
        "boundary_behavior",
        "验证 Requests 对 303/302/301 POST 的方法重写。",
        [evidence("sessions.py", 370, 392)],
        ["303", "302", "POST"],
        "Requests 遇到 301/302/303 都把所有方法改 GET 吗？请把 HEAD 例外和 POST+301 特例按 `_rebuild_method` 分支讲明白。",
        "Redirect method rewrite in Requests: 303, 302, POST+301 and HEAD exceptions—implementation, not generic RFC summary。",
    ),
    scenario(
        "rebuild-proxies",
        "data_flow",
        "覆盖 no_proxy、环境代理与 Proxy-Authorization 重建。",
        [evidence("sessions.py", 334, 368)],
        ["no_proxy", "resolve_proxies", "Proxy-Authorization"],
        "重定向后旧 Proxy-Authorization 会一直带着吗？环境变量、no_proxy 和代理 URL 自带账号时，headers 与代理映射怎样重算？",
        "`rebuild_proxies` after redirect: strip stale proxy auth, resolve environment/no_proxy, select scheme proxy, then conditionally add Basic auth。",
    ),
    scenario(
        "adapter-prefix",
        "boundary_behavior",
        "验证 Session.mount 的最长前缀优先与 get_adapter 选择。",
        [evidence("sessions.py", 870, 897)],
        ["mount", "prefix", "InvalidSchema"],
        "同时 mount `https://` 和 `https://api.example.com/`，是不是按注册先后选？说明 mount 如何排序，get_adapter 匹配不到又抛什么。",
        "Adapter routing uses URL prefixes：show case normalization, longest-prefix ordering after mount, and InvalidSchema fallback。",
    ),
    scenario(
        "prepare-url",
        "data_flow",
        "覆盖 URL 清理、非 HTTP scheme、IDNA、认证与 query 合并。",
        [evidence("models.py", 481, 561)],
        ["IDNA", "url", "params"],
        "URL 带中文域名、空白、user:pass 和已有 query，再传 params；prepare_url 是先 IDNA 还是先拼 query？像 mailto 这种 scheme 又会怎样？",
        "`PreparedRequest.prepare_url` branches: trim, non-HTTP bypass, parse/auth/IDNA, path normalization, existing query plus params。",
    ),
    scenario(
        "prepare-body",
        "boundary_behavior",
        "区分 JSON、流式 body、表单与 multipart 文件。",
        [evidence("models.py", 574, 650)],
        ["json", "Transfer-Encoding", "multipart"],
        "data 是 generator、json 也非空、files 还有内容时谁优先？Content-Length 算不出是不是直接 chunked；流式 body 能不能和 files 混用？",
        "Body preparation precedence: json vs stream-like data vs files/form, content length, Transfer-Encoding, rewind metadata and invalid stream+files。",
    ),
    scenario(
        "prepare-auth-cookies",
        "data_flow",
        "解释 auth 对 URL 的依赖、Content-Length 重算与 cookie header 一次性生成。",
        [evidence("models.py", 668, 718)],
        ["prepare_auth", "Content-Length", "Cookie"],
        "认证 callable 改了 body 后 Content-Length 会不会过期？Cookie header 后续调用 prepare_cookies 能否覆盖旧值；两个问题请分别沿代码回答。",
        "Auth may mutate request body, cookies are header-cached：explain auth extraction/call + length recompute, then one-shot Cookie preparation behavior。",
    ),
    scenario(
        "iter-content",
        "boundary_behavior",
        "覆盖流消费、缓存复用、异常映射与 unicode 解码。",
        [evidence("models.py", 905, 973)],
        ["iter_content", "StreamConsumedError", "decode_unicode"],
        "Response 已经消费过再 iter_content，是从缓存切片还是报 StreamConsumedError？chunk_size 类型错误、底层协议异常和 decode_unicode 各在哪处理？",
        "`Response.iter_content` state machine: raw streaming, exception translation, consumed/cache reuse, chunk validation and unicode decoding。",
    ),
    scenario(
        "iter-lines",
        "data_flow",
        "解释跨 chunk pending line 与自定义 delimiter。",
        [evidence("models.py", 976, 1028)],
        ["pending", "splitlines", "delimiter"],
        "一行被拆成两个网络 chunk，iter_lines 会不会重复或提前 yield？默认 splitlines 和自定义 delimiter 对 pending 判断有什么不同？",
        "Logical line framing over chunks: `iter_content` feed, pending tail merge, splitlines vs delimiter, final flush。",
    ),
    scenario(
        "response-text-json",
        "boundary_behavior",
        "区分 content、text 编码推断和 JSON 编码探测/异常包装。",
        [evidence("models.py", 1031, 1120)],
        ["encoding", "guess_json_utf", "JSONDecodeError"],
        "没声明 charset 的响应，`.text` 和 `.json()` 会不会用同一套猜测？空 body、UTF BOM、解码失败各怎样回退，异常类型是谁包装的？",
        "Compare Response.content/text/json: cache state, apparent encoding, `guess_json_utf`, fallback decoding and RequestsJSONDecodeError wrapping。",
    ),
    scenario(
        "adapter-tls-send",
        "cross_file_trace",
        "追踪 HTTPAdapter 的证书配置、请求 URL 选择与 urllib3 调用。",
        [
            evidence("adapters.py", 307, 363),
            evidence("adapters.py", 565, 597),
            evidence("adapters.py", 634, 748),
        ],
        ["cert_verify", "request_url", "urlopen"],
        "最终发请求前 verify/cert、proxy URL 形式、timeout 和 chunked 分散在好几处；请从 send 串到 cert_verify、request_url、conn.urlopen 及异常映射。",
        "Actual HTTPAdapter boundary: connection selection, TLS verify/cert, URL form for proxy, TimeoutSauce, urlopen and transport exception translation。",
    ),
]


def fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".md", ".toml"}
    ):
        rel = path.relative_to(root).as_posix().encode()
        digest.update(rel)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def expand(repository_id: str, specs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for spec in specs:
        for index, (language, question) in enumerate(spec["questions"], start=1):
            if language == "zh-en" and not any("\u4e00" <= char <= "\u9fff" for char in question):
                question = "我怀疑这里理解错了，请按源码核对：" + question
            suffix = "a" if index == 1 else "b"
            requirements = [
                {**item, "segments": chunk_compatible_segments(item)} for item in spec["evidence"]
            ]
            result.append(
                {
                    "id": f"m200-{repository_id}-{spec['slug']}-{suffix}",
                    "repository_id": repository_id,
                    "source_set": "master_200_new_real_repositories",
                    "scenario": spec["slug"],
                    "language": language,
                    "category": spec["category"],
                    "difficulty": "very_hard",
                    "protects": spec["protects"],
                    "challenge_features": spec["features"],
                    "input": {"question": question},
                    "expected": {
                        "outcome": "answered",
                        "evidence": [
                            segment
                            for requirement in requirements
                            for segment in requirement["segments"]
                        ],
                        "evidence_requirements": requirements,
                        "required_terms": spec["required_terms"],
                    },
                    "design_notes": [
                        "Question is newly written for the frozen local source version.",
                        "It intentionally combines ambiguity, noise or multiple reasoning demands.",
                    ],
                }
            )
    return result[:limit]


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    old_cases = []
    for original in source["cases"]:
        case = dict(original)
        case["repository_id"] = "sample_repo"
        case["source_set"] = "task11_multilingual_realistic_v2"
        old_cases.append(case)

    new_cases = (
        expand("httpx", HTTPX, 30) + expand("click", CLICK, 35) + expand("requests", REQUESTS, 35)
    )
    repositories = []
    for repository_id, metadata in REPOSITORIES.items():
        root = PROJECT_ROOT / metadata["local_root"]
        if not root.is_dir():
            raise FileNotFoundError(f"Missing frozen repository root: {root}")
        repositories.append(
            {"id": repository_id, **metadata, "source_fingerprint": fingerprint(root)}
        )

    cases = old_cases + new_cases
    counts = Counter(case["repository_id"] for case in cases)
    if counts != Counter({"sample_repo": 100, "httpx": 30, "click": 35, "requests": 35}):
        raise AssertionError(counts)
    if len({case["id"] for case in cases}) != 200:
        raise AssertionError("Case IDs must be unique")

    payload = {
        "schema_version": 1,
        "case_set": "codeinsight_master_200_final_capability",
        "description": "Task 11 v2 plus 100 newly authored, evidence-grounded Chinese and Chinese-English cases over three frozen real repositories.",
        "generation_contract": {
            "case_count": 200,
            "repository_counts": dict(counts),
            "new_case_languages": dict(Counter(case["language"] for case in new_cases)),
            "old_cases_policy": "Task 11 v2 questions and expected evidence are preserved; only repository_id and source_set provenance fields are added.",
            "external_repository_policy": "Pinned public source snapshots are read only; no upstream mutation or execution of analyzed repository code.",
        },
        "repositories": repositories,
        "cases": cases,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {TARGET} with {len(cases)} cases: {dict(counts)}")


if __name__ == "__main__":
    main()
