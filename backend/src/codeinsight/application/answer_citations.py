"""正文与引用的一致性检查。

为什么需要它：证据编号存在台账里，不等于引用支持结论。Q-012 阶段 C 的复测里出现过
「正文写对了 ``httpx/_client.py``，引用却挂到 ``_main.py`` / ``_api.py``」——编号映射是
直查、不存在错位，是模型挑错了编号。读者按引用去查是查不到出处的，所以这条一致性
本身就是质量边界。

检查只做确定性判断：正文点名的**台账内**文件，必须至少有一条引用落在同一个文件上。
它不是语义判断，也不试图评价解释对不对；判不准的时候宁可放过（不误伤正常答案）。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from codeinsight.domain.answer import AnswerCitation


def _mentioned(answer: str, path: str, basename_is_unique: bool) -> bool:
    normalized = answer.replace("\\", "/")
    if path in normalized:
        return True
    if not basename_is_unique:
        # 只在文件名唯一时才按简称判断，避免把 ``__init__.py`` 这类同名文件全判成不一致。
        return False
    basename = path.rsplit("/", 1)[-1]
    return basename in normalized


def uncited_mentioned_paths(
    answer: str,
    citations: Sequence[AnswerCitation],
    evidence_paths: Iterable[str],
) -> tuple[str, ...]:
    """返回正文点名、却没有任何引用覆盖的证据路径（保序去重）。"""

    normalized_paths = [path.replace("\\", "/") for path in evidence_paths]
    basenames = [path.rsplit("/", 1)[-1] for path in normalized_paths]
    cited = {citation.relative_path.replace("\\", "/") for citation in citations}
    flagged: list[str] = []
    for path, basename in zip(normalized_paths, basenames, strict=True):
        if path in cited or path in flagged:
            continue
        if _mentioned(answer, path, basenames.count(basename) == 1):
            flagged.append(path)
    return tuple(flagged)
