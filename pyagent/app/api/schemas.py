"""HTTP 契约 —— 与 Java 版 ``common/ApiResponse.java`` 完全同形。

前端与 Spring 转发层都按这个形状解析，所以 **字段名与默认值一个都不能改**：

.. code-block:: json

   {"code": 0, "message": "success", "data": {...}}

* ``ok(data)`` → ``code=0`` / ``message="success"``
* ``fail(code, message)`` → ``code=业务码`` / ``message=中文原因`` / ``data=null``

Java 全局处理器把 ``BusinessException`` 转成 **HTTP 400**（业务码放 body 里），
其它异常转成 **HTTP 500**。这里由 ``main.py`` 的异常处理器对齐。
"""

from __future__ import annotations

from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    code: int = 0
    message: str = "success"
    data: Optional[T] = None

    @classmethod
    def ok(cls, data: Optional[T] = None) -> "ApiResponse[T]":
        return cls(code=0, message="success", data=data)

    @classmethod
    def fail(cls, code: int, message: str) -> "ApiResponse[object]":
        return cls(code=code, message=message, data=None)
