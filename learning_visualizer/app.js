const PATHS = {
  frontend: "C:\\Users\\Alex\\Desktop\\workspace\\Project-0008-CodeInsight\\frontend\\src\\App.tsx",
  routes: "C:\\Users\\Alex\\Desktop\\workspace\\Project-0008-CodeInsight\\backend\\src\\codeinsight\\api\\routes.py",
  router: "C:\\Users\\Alex\\Desktop\\workspace\\Project-0008-CodeInsight\\backend\\src\\codeinsight\\application\\query_router.py",
  plan: "C:\\Users\\Alex\\Desktop\\workspace\\Project-0008-CodeInsight\\backend\\src\\codeinsight\\domain\\query_plan.py",
  autoAnswer: "C:\\Users\\Alex\\Desktop\\workspace\\Project-0008-CodeInsight\\backend\\src\\codeinsight\\application\\auto_answer_repository.py",
  search: "C:\\Users\\Alex\\Desktop\\workspace\\Project-0008-CodeInsight\\backend\\src\\codeinsight\\application\\search_repository.py",
  semantic: "C:\\Users\\Alex\\Desktop\\workspace\\Project-0008-CodeInsight\\backend\\src\\codeinsight\\retrieval\\semantic.py",
  hybrid: "C:\\Users\\Alex\\Desktop\\workspace\\Project-0008-CodeInsight\\backend\\src\\codeinsight\\retrieval\\hybrid.py",
  answer: "C:\\Users\\Alex\\Desktop\\workspace\\Project-0008-CodeInsight\\backend\\src\\codeinsight\\application\\answer_repository.py",
  prompt: "C:\\Users\\Alex\\Desktop\\workspace\\Project-0008-CodeInsight\\backend\\src\\codeinsight\\prompts\\code_answer.py",
  chat: "C:\\Users\\Alex\\Desktop\\workspace\\Project-0008-CodeInsight\\backend\\src\\codeinsight\\infrastructure\\openai_chat.py",
  embedding: "C:\\Users\\Alex\\Desktop\\workspace\\Project-0008-CodeInsight\\backend\\src\\codeinsight\\infrastructure\\embeddings.py",
  workflow: "C:\\Users\\Alex\\Desktop\\workspace\\Project-0008-CodeInsight\\backend\\src\\codeinsight\\agent\\workflow.py",
};

const COMMON_PACKET = {
  repository_root: "tests/fixtures/sample_repo",
  question: "checkout 如何校验输入？",
  limit: 5,
};

const MODES = {
  main: {
    title: "一次请求主链",
    description: "从前端问题开始，沿着当前 Auto Answer 主链观察对象如何变化。",
    canvas: { width: 930, height: 960 },
    steps: ["question", "client-auto", "client-post", "client-fetch", "api", "router", "plan", "route", "index", "subquestions", "citation", "response"],
    nodes: [
      {
        id: "question",
        x: 40,
        y: 36,
        stage: "input",
        title: "用户问题",
        subtitle: "frontend/src/App.tsx",
        summary: "前端收集仓库根目录、问题和 limit，并准备发起一次只读请求。",
        path: PATHS.frontend,
        function: "App.submit()",
        calls: ["autoAnswerRepository()", "postJson()", "fetch()"],
        input: COMMON_PACKET,
        output: { method: "POST", path: "/api/v1/auto/answer", body: COMMON_PACKET },
        failure: "输入为空、网络请求失败，或后端返回错误。",
        packetLabel: "请求 JSON",
        packet: COMMON_PACKET,
      },
      {
        id: "client-auto",
        x: 350,
        y: 36,
        stage: "input",
        title: "前端 API 封装",
        subtitle: "frontend/src/api/codeinsight.ts",
        summary: "把页面提交动作转换成 auto-answer 请求，并把 HTTP 响应交回 React。",
        path: "C:\\Users\\Alex\\Desktop\\workspace\\Project-0008-CodeInsight\\frontend\\src\\api\\codeinsight.ts",
        function: "autoAnswerRepository()",
        calls: ["postJson()"],
        input: COMMON_PACKET,
        output: { next: "postJson('/auto/answer', request)" },
        failure: "后端返回非 2xx 时抛出 Error，交给 App.submit() 的 catch。",
        packetLabel: "客户端请求",
        packet: { path: "/auto/answer", method: "POST", body: COMMON_PACKET },
      },
      {
        id: "client-post",
        x: 660,
        y: 36,
        stage: "input",
        title: "通用 JSON 请求函数",
        subtitle: "frontend/src/api/codeinsight.ts",
        summary: "序列化 request、设置 Content-Type，并调用浏览器的 fetch()。",
        path: "C:\\Users\\Alex\\Desktop\\workspace\\Project-0008-CodeInsight\\frontend\\src\\api\\codeinsight.ts",
        function: "postJson()",
        calls: ["fetch()", "response.json()"],
        input: { path: "/auto/answer", request: COMMON_PACKET },
        output: { response: "AutoAnswerResponse JSON" },
        failure: "网络失败或 response.ok 为 false 时抛出错误。",
        packetLabel: "HTTP JSON",
        packet: { headers: { "Content-Type": "application/json" }, body: COMMON_PACKET },
      },
      {
        id: "client-fetch",
        x: 660,
        y: 220,
        stage: "input",
        title: "浏览器 HTTP 调用",
        subtitle: "Web API · 不属于项目源码",
        summary: "浏览器把 POST 请求发送到后端的 /api/v1/auto/answer 路由。",
        path: "浏览器内置 Web API（无项目文件）",
        function: "fetch()",
        calls: ["POST /api/v1/auto/answer"],
        input: { method: "POST", url: "/api/v1/auto/answer" },
        output: { enters: "routes.py::auto_answer()" },
        failure: "网络不可达、CORS 或后端未启动时请求失败。",
        packetLabel: "HTTP 边界",
        packet: { method: "POST", path: "/api/v1/auto/answer", body: COMMON_PACKET },
      },
      {
        id: "api",
        x: 350,
        y: 220,
        stage: "input",
        title: "FastAPI 路由入口",
        subtitle: "api/routes.py · POST /api/v1/auto/answer",
        summary: "HTTP 层只负责接收请求、创建模型适配器、选择路线并转换响应。",
        path: PATHS.routes,
        function: "auto_answer()",
        calls: ["route_question()", "auto_answer_repository()", "_auto_response()"],
        input: COMMON_PACKET,
        output: { router: "QueryRouterResult", next: "linear or agent" },
        failure: "模型未配置：503；模型调用或响应解析失败：502；参数或仓库错误：400。",
        packetLabel: "HTTP 边界",
        packet: { request: COMMON_PACKET, response_type: "AutoAnswerResponse" },
      },
      {
        id: "router",
        x: 40,
        y: 220,
        stage: "application",
        title: "Router 规划",
        subtitle: "application/query_router.py",
        summary: "把自然语言问题交给模型规划，但不让模型直接决定文件路径或证据。",
        path: PATHS.router,
        function: "route_question()",
        calls: ["build_router_prompt()", "OpenAIChatModel.complete()", "parse_query_plan()"],
        input: { original_question: COMMON_PACKET.question, model: "Chat completion" },
        output: { normalized_question: "checkout 输入校验", subquestions: 1, confidence: 0.92 },
        failure: "模型返回非 JSON、字段不完整、路由或检索模式不受支持时使用 fallback。",
        packetLabel: "Router 计划草稿",
        packet: { normalized_question: "checkout 输入校验", execution_route: "linear", retrieval_mode: "bm25" },
      },
      {
        id: "plan",
        x: 40,
        y: 404,
        stage: "application",
        title: "解析 Router JSON",
        subtitle: "application/query_router.py",
        summary: "把 Router 的结果变成经过严格校验的计划，限制子问题数量和执行路线。",
        path: PATHS.router,
        function: "parse_query_plan()",
        calls: ["_decode_json_object()", "QueryPlan.__post_init__()"],
        input: { subquestions: ["checkout 输入校验"], execution_route: "linear" },
        output: { subquestions: 1, retrieval_modes: ["bm25"], execution_route: "linear" },
        failure: "confidence 不在 0 到 1、子问题超过 8 个、模式不匹配都会失败。",
        packetLabel: "QueryPlan",
        packet: { subquestions: [{ question: "checkout 输入校验", retrieval_mode: "bm25" }], execution_route: "linear" },
      },
      {
        id: "index",
        x: 660,
        y: 404,
        stage: "retrieval",
        title: "建立共享语义索引",
        subtitle: "application/search_repository.py",
        summary: "扫描仓库并加载或增量构建语义索引，不为每个子问题重复向量化代码。",
        path: PATHS.search,
        function: "build_repository_semantic_index()",
        calls: ["scan_repository()", "build_persistent_semantic_index()", "OpenAIEmbeddingModel.embed()"],
        input: { repository_root: COMMON_PACKET.repository_root, chunk_max_lines: 80 },
        output: { index: "SemanticIndex", cache: "本地 JSON", reused_code_vectors: true },
        failure: "Embedding 服务未配置、返回数量或维度不正确、仓库没有代码块。",
        packetLabel: "SemanticIndex",
        packet: { entries: "源码代码块", model: "OpenAI-compatible embedding", shared: true, cache: "persistent local JSON" },
      },
      {
        id: "subquestions",
        x: 660,
        y: 588,
        stage: "retrieval",
        title: "检索一个子问题",
        subtitle: "application/search_repository.py",
        summary: "逐个子问题执行 BM25 + Semantic，融合后保存自己的 Top-K 结果和证据编号。",
        path: PATHS.search,
        function: "retrieve_subquestion_evidence()",
        calls: ["candidate_retrieval_modes()", "search_repository()", "search_chunks_semantic()", "rerank_ranked_chunks()"],
        input: { subquestions: ["checkout 输入校验"], semantic_index: "shared" },
        output: { evidence_groups: 1, each_limit: 5, evidence_ids: ["E1", "E2"] },
        failure: "没有检索结果时，该子问题变成 insufficient_evidence，不会让模型凭空补答案。",
        packetLabel: "SubQuestionEvidence",
        packet: { question: "checkout 输入校验", results: "RankedChunk × 5", evidence_ids: ["E1", "E2"] },
      },
      {
        id: "route",
        x: 350,
        y: 404,
        stage: "application",
        title: "按子问题编排回答",
        subtitle: "application/auto_answer_repository.py",
        summary: "linear 直接逐题生成；agent 进入 LangGraph 的 Draft、Critic、Reviser 流程。",
        path: PATHS.autoAnswer,
        function: "auto_answer_repository()",
        calls: ["build_repository_semantic_index()", "retrieve_subquestion_evidence()", "build_answer_prompt()", "OpenAIChatModel.generate()", "map_model_answer()"],
        input: { execution_route: "linear", evidence_groups: 1 },
        output: { next: "answer generation" },
        failure: "insufficient 路线会停止生成；Agent 还有有限次数的修订边界。",
        packetLabel: "执行选择",
        packet: { route: "linear", reason: "当前示例证据条件满足" },
      },
      {
        id: "citation",
        x: 350,
        y: 588,
        stage: "validation",
        title: "回答与引用映射",
        subtitle: "application/answer_repository.py",
        summary: "模型只能返回证据 ID，后端把证据 ID 映射成真实文件路径和 1-based 行号。",
        path: PATHS.answer,
        function: "map_model_answer()",
        calls: ["AnswerCitation(...)"],
        input: { answer: "模型结构化回答", evidence_ids: ["E1"] },
        output: { outcome: "answered", citations: [{ evidence_id: "E1", path: "src/checkout.py" }] },
        failure: "未知证据 ID、回答与证据不匹配、引用边界不合法时拒绝或降级。",
        packetLabel: "Citation 映射",
        packet: { model_citations: ["E1"], local_citation: { path: "src/checkout.py", start_line: 10, end_line: 25 } },
      },
      {
        id: "response",
        x: 40,
        y: 772,
        stage: "validation",
        title: "最终响应",
        subtitle: "api/routes.py",
        summary: "把计划、子问题答案、引用和用量汇总成前端可以展示的响应。",
        path: PATHS.routes,
        function: "_auto_response()",
        calls: ["_citation_responses()", "AutoAnswerResponse(...)"],
        input: { subquestion_answers: 1, citations: 1 },
        output: { outcome: "answered", answer: "带路径和行号的回答" },
        failure: "应用异常会在 HTTP 层转换成明确的 400、502 或 503。",
        packetLabel: "前端响应",
        packet: { outcome: "answered", answer: "checkout 的输入校验位于 src/checkout.py", citations: ["E1"] },
      },
    ],
    edges: [
      { from: "question", to: "client-auto", label: "调用" },
      { from: "client-auto", to: "client-post", label: "调用" },
      { from: "client-post", to: "client-fetch", label: "调用" },
      { from: "client-fetch", to: "api", label: "HTTP POST", kind: "http" },
      { from: "api", to: "router", label: "调用" },
      { from: "router", to: "plan", label: "调用" },
      { from: "plan", to: "route", label: "返回 QueryPlan", kind: "return" },
      { from: "route", to: "index", label: "调用" },
      { from: "index", to: "subquestions", label: "随后调用 retrieve()", kind: "data" },
      { from: "subquestions", to: "citation", label: "返回证据", kind: "return" },
      { from: "citation", to: "response", label: "调用" },
    ],
  },
  retrieval: {
    title: "检索放大",
    description: "只放大一个子问题，看 BM25、Semantic、RRF 和代码感知重排如何协作。",
    canvas: { width: 930, height: 1140 },
    steps: ["retrieval-question", "bm25", "semantic", "fusion", "code-rerank", "top-k", "evidence"],
    nodes: [
      {
        id: "retrieval-question",
        x: 350,
        y: 36,
        stage: "input",
        title: "一个子问题",
        subtitle: "application/search_repository.py",
        summary: "检索阶段不处理整个用户问题，只处理 QueryPlan 拆出的一个独立子问题。",
        path: PATHS.search,
        function: "retrieve_subquestion_evidence()",
        calls: ["candidate_retrieval_modes()", "search_repository()", "search_chunks_semantic()", "rerank_ranked_chunks()"],
        input: { question: "checkout 如何校验输入？", limit: 5 },
        output: { candidate_limit: 20, modes: ["bm25", "semantic"] },
        failure: "limit 非正数或 primary_mode 不受支持时直接报错。",
        packetLabel: "检索输入",
        packet: { question: "checkout 如何校验输入？", candidate_limit: 20, final_limit: 5 },
      },
      {
        id: "bm25",
        x: 40,
        y: 230,
        stage: "retrieval",
        title: "BM25 稀疏分支",
        subtitle: "retrieval/bm25.py",
        summary: "用词面匹配快速找函数名、文件名、路径和代码标识符。",
        path: "C:\\Users\\Alex\\Desktop\\workspace\\Project-0008-CodeInsight\\backend\\src\\codeinsight\\retrieval\\bm25.py",
        function: "search_chunks_bm25()",
        calls: ["tokenize_query()", "_body_score()", "_field_bonuses()"],
        input: { question: "checkout 如何校验输入？", chunks: "源码代码块" },
        output: { count: 20, reason: "direct_match", ranks: [1, 2, 3] },
        failure: "词面没有命中时，结果可能为空或缺少语义改写。",
        packetLabel: "BM25 候选",
        packet: { source: "bm25", candidates: 20, example: "src/checkout.py:10-25" },
      },
      {
        id: "semantic",
        x: 660,
        y: 230,
        stage: "retrieval",
        title: "Semantic 向量分支",
        subtitle: "retrieval/semantic.py",
        summary: "把问题变成 query vector，与共享索引中的代码向量计算余弦相似度。",
        path: PATHS.semantic,
        function: "search_chunks_semantic()",
        calls: ["OpenAIEmbeddingModel.embed()", "_cosine_similarity()"],
        input: { query: "checkout 如何校验输入？", index: "SemanticIndex" },
        output: { count: 20, reason: "semantic_match", score: "cosine similarity" },
        failure: "查询向量和索引维度不匹配，或 Embedding 服务返回非法向量。",
        packetLabel: "Semantic 候选",
        packet: { source: "semantic", candidates: 20, query_vector: "[... ]", example: "src/validators.py:1-20" },
      },
      {
        id: "fusion",
        x: 350,
        y: 420,
        stage: "retrieval",
        title: "公开重排入口",
        subtitle: "retrieval/hybrid.py",
        summary: "不直接相加 BM25 和 cosine 原始分数，而是按各自排名计算 RRF，并记录多来源一致性。",
        path: PATHS.hybrid,
        function: "rerank_ranked_chunks()",
        calls: ["_rerank()"],
        input: { bm25: "Top-20", semantic: "Top-20", rrf_k: 60 },
        output: { unique_candidates: 31, agreement: "bm25 + semantic" },
        failure: "没有来源时返回空结果；limit 非正数时抛出 ValueError。",
        packetLabel: "融合候选",
        packet: { rrf: "weight / (60 + rank)", weights: { bm25: 1, semantic: 0.8 }, agreement: "same path + line range" },
      },
      {
        id: "code-rerank",
        x: 350,
        y: 610,
        stage: "retrieval",
        title: "内部融合与排序",
        subtitle: "retrieval/hybrid.py",
        summary: "用函数名、路径、正文词重叠和多来源一致性做可解释的最终排序。",
        path: PATHS.hybrid,
        function: "_rerank()",
        calls: ["_chunk_key()", "_code_relevance()", "sorted(...)"],
        input: { candidates: "融合后的候选", query_tokens: "checkout / validate" },
        output: { score: "RRF × 100 + code relevance + agreement" },
        failure: "它不能创造新证据，也不能修复前面完全没有召回的代码块。",
        packetLabel: "重排分数",
        packet: { symbol_overlap: 0.12, path_overlap: 0.08, body_overlap: 0.05, model_call: false },
      },
      {
        id: "top-k",
        x: 350,
        y: 800,
        stage: "retrieval",
        title: "计算代码相关性",
        subtitle: "纯 Python，不调用 LLM",
        summary: "从中间候选池中截取最终交给回答模型的代码块。",
        path: PATHS.search,
        function: "_code_relevance()",
        calls: ["tokenize_query()", "tokenize_terms()", "tokenize_path()"],
        input: { candidates: "最多 40 个来源结果" },
        output: { selected: 5, type: "tuple[RankedChunk, ...]" },
        failure: "Top-K 太小可能漏掉必要证据；Top-K 太大则增加回答上下文。",
        packetLabel: "RankedChunk Top-K",
        packet: { rank: 1, path: "src/checkout.py", start_line: 10, end_line: 25, reason: "hybrid_match" },
      },
      {
        id: "evidence",
        x: 350,
        y: 990,
        stage: "validation",
        title: "返回最终 Top-K",
        subtitle: "tuple[RankedChunk, ...]",
        summary: "给这个子问题保存独立的结果和证据 ID，不能与其他子问题混用。",
        path: "C:\\Users\\Alex\\Desktop\\workspace\\Project-0008-CodeInsight\\backend\\src\\codeinsight\\domain\\retrieval.py",
        function: "retrieve_subquestion_evidence()",
        calls: ["return tuple[RankedChunk, ...]"],
        input: { results: "RankedChunk × 5" },
        output: { evidence_ids: ["E1", "E2"], covered: true },
        failure: "results 数量和 evidence_ids 数量不一致时拒绝构造对象。",
        packetLabel: "SubQuestionEvidence",
        packet: { question: "checkout 如何校验输入？", evidence_ids: ["E1", "E2"], covered: true },
      },
    ],
    edges: [
      { from: "retrieval-question", to: "bm25", label: "调用 BM25" },
      { from: "retrieval-question", to: "semantic", label: "调用 Semantic" },
      { from: "bm25", to: "fusion", label: "Top-20" },
      { from: "semantic", to: "fusion", label: "Top-20" },
      { from: "fusion", to: "code-rerank", label: "调用内部重排" },
      { from: "code-rerank", to: "top-k", label: "调用特征评分" },
      { from: "top-k", to: "evidence", label: "返回 Top-K", kind: "return" },
    ],
  },
  agent: {
    title: "Agent 循环",
    description: "观察证据已经找到以后，Agent 如何起草、审查、局部修订并在上限内结束。",
    canvas: { width: 930, height: 1060 },
    steps: ["agent-retrieve", "draft", "review", "gate", "agent-review", "agent-gate", "revise", "finalize", "agent-response"],
    nodes: [
      {
        id: "agent-retrieve",
        x: 350,
        y: 36,
        stage: "retrieval",
        title: "启动 Agent 工作流",
        subtitle: "agent/workflow.py",
        summary: "Agent 也从检索开始；它不能凭空审查没有证据支持的答案。",
        path: PATHS.workflow,
        function: "run_citation_agent()",
        calls: ["build_repository_semantic_index()", "retrieve()", "StateGraph.compile()"],
        input: { query_plan: "QueryPlan", repository: COMMON_PACKET.repository_root },
        output: { evidence_groups: 1, results: "Top-K per subquestion" },
        failure: "没有证据时走 no_evidence，直接结束，不进入无限生成循环。",
        packetLabel: "Agent State",
        packet: { phase: "retrieve", evidence_groups: 1, revisions: 0 },
      },
      {
        id: "draft",
        x: 350,
        y: 190,
        stage: "retrieval",
        title: "检索节点",
        subtitle: "LangGraph node",
        summary: "读取 Agent State；有 QueryPlan 时调用跨子问题检索，否则直接调用 search_repository()。",
        path: PATHS.workflow,
        function: "retrieve()",
        calls: ["_retrieve_query_plan()", "search_repository()"],
        input: { query_plan: "QueryPlan", repository: COMMON_PACKET.repository_root },
        output: { evidence_groups: 1, results: "Top-K per subquestion" },
        failure: "没有证据时走 no_evidence，直接结束，不进入生成循环。",
        packetLabel: "Retrieve State",
        packet: { phase: "retrieve", evidence_groups: 1, revisions: 0 },
      },
      {
        id: "review",
        x: 350,
        y: 344,
        stage: "retrieval",
        title: "跨子问题证据检索",
        subtitle: "agent/workflow.py",
        summary: "把 QueryPlan 中的每个子问题交给检索函数，并为每题保留自己的证据组。",
        path: PATHS.workflow,
        function: "_retrieve_query_plan()",
        calls: ["retrieve_subquestion_evidence()", "_map_subquestion_answers()"],
        input: { query_plan: "QueryPlan", subquestions: 1 },
        output: { evidence_groups: 1, results: "Top-K per subquestion" },
        failure: "没有证据时会进入 no_evidence 分支，不会凭空生成答案。",
        packetLabel: "SubQuestionEvidence",
        packet: { question: "checkout 如何校验输入？", evidence_ids: ["E1", "E2"] },
      },
      {
        id: "gate",
        x: 40,
        y: 520,
        stage: "model",
        title: "逐题起草答案",
        subtitle: "LangGraph node",
        summary: "每个子问题使用自己的证据生成局部答案，再按 QueryPlan 顺序汇总。",
        path: PATHS.workflow,
        function: "draft()",
        calls: ["build_answer_prompt()", "OpenAIChatModel.complete()", "_validate_local_citations()"],
        input: { evidence: "SubQuestionEvidence", prompt: "code answer prompt" },
        output: { outcome: "answered", citations: ["E1"] },
        failure: "模型输出结构不合法，或回答没有使用允许的证据 ID。",
        packetLabel: "Draft",
        packet: { phase: "draft", answer: "checkout 使用 validator 校验输入", citations: ["E1"] },
      },
      {
        id: "agent-review",
        x: 350,
        y: 520,
        stage: "model",
        title: "审查引用和回答",
        subtitle: "LangGraph node",
        summary: "审查草稿是否有足够证据、是否引用了错误证据，以及回答是否符合要求。",
        path: PATHS.workflow,
        function: "review()",
        calls: ["build_review_prompt()", "parse_citation_review()"],
        input: { draft: "ModelAnswer", evidence: "local evidence group" },
        output: { passed: false, feedback: "E1 不足以支持第二个结论" },
        failure: "Critic 只能审查已有证据，不能重新检索补齐缺失证据。",
        packetLabel: "CriticResult",
        packet: { passed: false, supported_evidence_ids: ["E1"], feedback: "需要收窄答案" },
      },
      {
        id: "agent-gate",
        x: 660,
        y: 520,
        stage: "application",
        title: "决定下一步",
        subtitle: "conditional edge",
        summary: "根据审查结果决定结束、修订，或在证据不足时冻结当前子问题。",
        path: PATHS.workflow,
        function: "route_review()",
        calls: ["finalize_reviewed()", "revise()", "finalize_revised()"],
        input: { passed: false, revisions: 1, max_revisions: 5 },
        output: { next: "revise" },
        failure: "达到最多 5 次修订后必须结束，不能无限循环。",
        packetLabel: "路由决定",
        packet: { condition: "passed == false", next: "revise", remaining: 4 },
      },
      {
        id: "revise",
        x: 660,
        y: 710,
        stage: "model",
        title: "局部修订失败项",
        subtitle: "LangGraph node",
        summary: "把 Critic 反馈交给模型，只对未通过的局部答案进行修订。",
        path: PATHS.workflow,
        function: "revise()",
        calls: ["build_revision_prompt()", "OpenAIChatModel.complete()", "_validate_local_citations()"],
        input: { draft: "failed subquestion draft", feedback: "CriticResult" },
        output: { revised_draft: "new ModelAnswer", revisions: 2 },
        failure: "修订不会重新检索；如果根因是召回不足，它无法凭空创造证据。",
        packetLabel: "Revision State",
        packet: { phase: "revise", revised: true, revisions: 2, retrieval_repeated: false },
      },
      {
        id: "finalize",
        x: 350,
        y: 710,
        stage: "validation",
        title: "结束循环并汇总",
        subtitle: "agent/workflow.py",
        summary: "把通过或达到修订上限的结果冻结，按照子问题顺序确定性汇总。",
        path: PATHS.workflow,
        function: "finalize_revised()",
        calls: ["_agent_result()", "_combine_subquestion_drafts()"],
        input: { drafts: "subquestion drafts", reviews: "review results" },
        output: { revisions: 2, final_outcome: "answered" },
        failure: "局部引用仍需经过代码边界和 evidence ID 校验。",
        packetLabel: "AgentResult",
        packet: { outcome: "answered", revisions: 2, subquestions: 1, citations: ["E1"] },
      },
      {
        id: "agent-response",
        x: 350,
        y: 900,
        stage: "validation",
        title: "转换为公开响应",
        subtitle: "api/routes.py",
        summary: "最终返回答案、引用、修订次数和可供学习的 Agent 事件摘要。",
        path: PATHS.routes,
        function: "_auto_from_agent()",
        calls: ["_citation_responses()", "AgentAnswerResponse(...)"],
        input: { agent_result: "AgentResult" },
        output: { response: "AutoAnswerResponse", events: ["retrieve", "draft", "review", "revise"] },
        failure: "HTTP 层仍然会把模型和应用错误转换成公开的错误响应。",
        packetLabel: "Agent 事件",
        packet: { sequence: 4, step: "revise", summary: "已局部修订一个未通过子问题" },
      },
    ],
    edges: [
      { from: "agent-retrieve", to: "draft", label: "调用 retrieve()" },
      { from: "draft", to: "review", label: "调用 _retrieve_query_plan()" },
      { from: "review", to: "gate", label: "返回证据", kind: "return" },
      { from: "gate", to: "agent-review", label: "调用 review()" },
      { from: "agent-review", to: "agent-gate", label: "返回审查结果", kind: "return" },
      { from: "agent-gate", to: "finalize", label: "通过 / 达到上限" },
      { from: "agent-gate", to: "revise", label: "未通过" },
      { from: "revise", to: "agent-review", label: "再次调用 review()", kind: "loop" },
      { from: "finalize", to: "agent-response", label: "调用 _auto_from_agent()" },
    ],
  },
};

const state = {
  mode: "main",
  stepIndex: 0,
  selectedNodeId: "question",
  timer: null,
};

const elements = {
  canvas: document.querySelector("#graph-canvas"),
  edgeLayer: document.querySelector("#edge-layer"),
  nodeLayer: document.querySelector("#node-layer"),
  modeTitle: document.querySelector("#mode-title"),
  modeDescription: document.querySelector("#mode-description"),
  stepCounter: document.querySelector("#step-counter"),
  stepDescription: document.querySelector("#step-description"),
  progressFill: document.querySelector("#progress-fill"),
  detailStage: document.querySelector("#detail-stage"),
  detailTitle: document.querySelector("#detail-title"),
  detailSummary: document.querySelector("#detail-summary"),
  detailPath: document.querySelector("#detail-path"),
  detailFunction: document.querySelector("#detail-function"),
  detailCalls: document.querySelector("#detail-calls"),
  detailInput: document.querySelector("#detail-input"),
  detailOutput: document.querySelector("#detail-output"),
  detailFailure: document.querySelector("#detail-failure"),
  packetTitle: document.querySelector("#packet-title"),
  packetView: document.querySelector("#packet-view"),
  playButton: document.querySelector('[data-action="play"]'),
};

function currentMode() {
  return MODES[state.mode];
}

function currentNode() {
  return currentMode().nodes.find((node) => node.id === state.selectedNodeId) || currentMode().nodes[0];
}

function formatJson(value) {
  return JSON.stringify(value, null, 2);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatFunctionName(value) {
  return escapeHtml(value).replaceAll("_", "_<wbr>");
}

function shortPath(path) {
  const parts = String(path).split("\\");
  return parts.length > 2 ? `…\\${parts.slice(-2).join("\\")}` : path;
}

function renderMode() {
  const mode = currentMode();
  elements.modeTitle.textContent = mode.title;
  elements.modeDescription.textContent = mode.description;
  elements.canvas.style.width = `${mode.canvas.width}px`;
  elements.canvas.style.height = `${mode.canvas.height}px`;
  elements.nodeLayer.replaceChildren();
  elements.edgeLayer.replaceChildren();
  state.selectedNodeId = mode.steps[0];

  mode.nodes.forEach((node) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `flow-node stage-${node.stage}`;
    button.dataset.nodeId = node.id;
    button.style.left = `${node.x}px`;
    button.style.top = `${node.y}px`;
    button.innerHTML = `
      <span class="node-stage">${escapeHtml(node.stage)}</span>
      <code class="node-function">${formatFunctionName(node.function)}</code>
      <span class="node-title">${escapeHtml(node.title)}</span>
      <span class="node-subtitle">${escapeHtml(node.subtitle)}</span>
    `;
    button.addEventListener("click", () => selectNode(node.id));
    elements.nodeLayer.append(button);
  });

  updateProgress();
  selectNode(state.selectedNodeId);
  window.requestAnimationFrame(drawEdges);
}

function drawEdges() {
  const mode = currentMode();
  const canvasRect = elements.canvas.getBoundingClientRect();
  elements.edgeLayer.setAttribute("width", String(canvasRect.width));
  elements.edgeLayer.setAttribute("height", String(canvasRect.height));
  elements.edgeLayer.setAttribute("viewBox", `0 0 ${canvasRect.width} ${canvasRect.height}`);
  elements.edgeLayer.replaceChildren();

  const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
  marker.setAttribute("id", "arrow");
  marker.setAttribute("markerWidth", "8");
  marker.setAttribute("markerHeight", "8");
  marker.setAttribute("refX", "7");
  marker.setAttribute("refY", "4");
  marker.setAttribute("orient", "auto");
  const arrowPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
  arrowPath.setAttribute("d", "M 0 0 L 8 4 L 0 8 z");
  arrowPath.setAttribute("fill", "#93a4b8");
  marker.append(arrowPath);
  elements.edgeLayer.append(marker);

  mode.edges.forEach((edge) => {
    const source = elements.nodeLayer.querySelector(`[data-node-id="${edge.from}"]`);
    const target = elements.nodeLayer.querySelector(`[data-node-id="${edge.to}"]`);
    if (!source || !target) return;

    const sourceRect = source.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    const sourceLeft = sourceRect.left - canvasRect.left;
    const sourceTop = sourceRect.top - canvasRect.top;
    const targetLeft = targetRect.left - canvasRect.left;
    const targetTop = targetRect.top - canvasRect.top;
    const sourceCenterX = sourceLeft + sourceRect.width / 2;
    const targetCenterX = targetLeft + targetRect.width / 2;
    const sourceCenterY = sourceTop + sourceRect.height / 2;
    const targetCenterY = targetTop + targetRect.height / 2;
    const isVertical = Math.abs(targetCenterY - sourceCenterY) > 18;
    let startX;
    let startY;
    let endX;
    let endY;
    let pathData;
    let labelX;
    let labelY;

    if (isVertical) {
      const goesDown = targetCenterY > sourceCenterY;
      startX = sourceCenterX;
      startY = sourceTop + (goesDown ? sourceRect.height : 0);
      endX = targetCenterX;
      endY = targetTop + (goesDown ? 0 : targetRect.height);
      const middleY = (startY + endY) / 2;
      pathData = `M ${startX} ${startY} C ${startX} ${middleY}, ${endX} ${middleY}, ${endX} ${endY}`;
      labelX = (startX + endX) / 2;
      labelY = middleY - 8;
    } else {
      const goesRight = targetCenterX >= sourceCenterX;
      startX = sourceLeft + (goesRight ? sourceRect.width : 0);
      startY = sourceCenterY;
      endX = targetLeft + (goesRight ? 0 : targetRect.width);
      endY = targetCenterY;
      const middleX = (startX + endX) / 2;
      pathData = `M ${startX} ${startY} C ${middleX} ${startY}, ${middleX} ${endY}, ${endX} ${endY}`;
      labelX = middleX;
      labelY = (startY + endY) / 2 - 8;
    }

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.classList.add("flow-edge");
    path.classList.add(`kind-${edge.kind || "call"}`);
    const sourceStep = mode.steps.indexOf(edge.from);
    const targetStep = mode.steps.indexOf(edge.to);
    if (sourceStep < state.stepIndex && targetStep <= state.stepIndex) path.classList.add("is-done");
    if (sourceStep === state.stepIndex || targetStep === state.stepIndex) path.classList.add("is-active");
    path.setAttribute("d", pathData);
    path.setAttribute("marker-end", "url(#arrow)");
    elements.edgeLayer.append(path);

    if (edge.label) {
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.classList.add("edge-label");
      label.setAttribute("x", String(labelX));
      label.setAttribute("y", String(labelY));
      label.textContent = edge.label;
      elements.edgeLayer.append(label);
    }
  });
}

function selectNode(nodeId) {
  state.selectedNodeId = nodeId;
  const node = currentNode();
  const outgoingCalls = currentMode().edges
    .filter((edge) => edge.from === node.id)
    .map((edge) => {
      const target = currentMode().nodes.find((item) => item.id === edge.to);
      return target ? `下一步：${target.function}（${edge.label}）` : null;
    })
    .filter(Boolean);
  const internalCalls = (node.calls || []).map((call) => `内部：${call}`);
  const callLines = [...internalCalls, ...outgoingCalls];
  document.querySelectorAll(".flow-node").forEach((button) => {
    button.classList.toggle("is-selected", button.dataset.nodeId === node.id);
  });
  elements.detailStage.textContent = node.stage;
  elements.detailTitle.textContent = node.title;
  elements.detailSummary.textContent = node.summary;
  elements.detailPath.textContent = node.path;
  elements.detailFunction.textContent = node.function;
  elements.detailCalls.textContent = callLines.length ? callLines.join("\n") : "—";
  elements.detailInput.textContent = formatJson(node.input);
  elements.detailOutput.textContent = formatJson(node.output);
  elements.detailFailure.textContent = node.failure;
  elements.packetTitle.textContent = node.packetLabel;
  elements.packetView.textContent = formatJson(node.packet);
}

function updateProgress() {
  const mode = currentMode();
  const currentStep = mode.steps[state.stepIndex];
  const node = mode.nodes.find((item) => item.id === currentStep) || mode.nodes[0];
  const lastIndex = Math.max(1, mode.steps.length - 1);
  elements.stepCounter.textContent = `第 ${state.stepIndex + 1} / ${mode.steps.length} 步`;
  elements.progressFill.style.width = `${(state.stepIndex / lastIndex) * 100}%`;
  elements.stepDescription.textContent = `当前：${node.title}。${node.summary}`;

  document.querySelectorAll(".flow-node").forEach((button) => {
    const step = mode.steps.indexOf(button.dataset.nodeId);
    button.classList.toggle("is-active", step === state.stepIndex);
    button.classList.toggle("is-done", step >= 0 && step < state.stepIndex);
    button.classList.toggle("is-future", step > state.stepIndex);
  });
  window.requestAnimationFrame(drawEdges);
}

function goToStep(nextIndex) {
  const mode = currentMode();
  state.stepIndex = Math.max(0, Math.min(nextIndex, mode.steps.length - 1));
  state.selectedNodeId = mode.steps[state.stepIndex];
  updateProgress();
  selectNode(state.selectedNodeId);
  if (state.stepIndex === mode.steps.length - 1) stopPlayback();
}

function startPlayback() {
  if (state.timer) return;
  elements.playButton.textContent = "暂停";
  elements.playButton.setAttribute("aria-pressed", "true");
  state.timer = window.setInterval(() => {
    if (state.stepIndex >= currentMode().steps.length - 1) {
      stopPlayback();
      return;
    }
    goToStep(state.stepIndex + 1);
  }, 1500);
}

function stopPlayback() {
  if (state.timer) window.clearInterval(state.timer);
  state.timer = null;
  elements.playButton.textContent = "播放";
  elements.playButton.setAttribute("aria-pressed", "false");
}

function setMode(modeKey) {
  if (!MODES[modeKey]) return;
  stopPlayback();
  state.mode = modeKey;
  state.stepIndex = 0;
  document.querySelectorAll(".mode-tab").forEach((button) => {
    const selected = button.dataset.mode === modeKey;
    button.classList.toggle("is-selected", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  renderMode();
}

document.querySelectorAll(".mode-tab").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

document.querySelector('[data-action="reset"]').addEventListener("click", () => goToStep(0));
document.querySelector('[data-action="previous"]').addEventListener("click", () => goToStep(state.stepIndex - 1));
document.querySelector('[data-action="next"]').addEventListener("click", () => goToStep(state.stepIndex + 1));
elements.playButton.addEventListener("click", () => {
  if (state.timer) stopPlayback();
  else startPlayback();
});

window.addEventListener("resize", () => window.requestAnimationFrame(drawEdges));
renderMode();
