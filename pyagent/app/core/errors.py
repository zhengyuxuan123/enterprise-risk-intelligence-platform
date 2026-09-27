"""错误类型：与 Java 版 ``com.example.riskplatform.common`` 保持一致。

Java 侧的契约是：

* ``BusinessException`` —— 默认 code=400。全局处理器把它转成 **HTTP 400** +
  ``ApiResponse.fail(code, message)``（注意：HTTP 状态码固定 400，业务码在 body 里）。
* 其它异常 —— HTTP 500 + ``ApiResponse.fail(500, message)``。

Python 版必须保持同一形状，否则 Spring 转发层与前端都会读错。
"""

from __future__ import annotations


class BusinessError(Exception):
    """业务异常。**不**表示 HTTP 层的错误码，只带业务码与可直接展示的中文原因。"""

    def __init__(self, message: str, code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - 便于日志
        return self.message


class NotConfiguredError(BusinessError):
    """能力未装配/未配置。

    这类错误一律**伴随一句可操作的说明**（该设哪个环境变量、或者该改哪个开关），
    而不是抛一句 "not configured" —— 这是 Java 侧一路坚持的写法，
    也是"改了代码却像没生效"这类问题最容易卡住人的地方。
    """
