"""仓库扫描和模型调用使用的领域异常。"""


class RepositoryScanError(ValueError):
    """仓库扫描无法启动时抛出。"""


class ModelConfigurationError(ValueError):
    """缺少运行时所需模型配置时抛出。"""


class ModelResponseError(ValueError):
    """模型输出无法转换成有证据支撑的答案时抛出。"""


class ModelCallError(RuntimeError):
    """已配置的模型请求失败时抛出。"""
