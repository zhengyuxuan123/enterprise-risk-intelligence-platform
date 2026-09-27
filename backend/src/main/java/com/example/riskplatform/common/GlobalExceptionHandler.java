package com.example.riskplatform.common;

import jakarta.servlet.http.HttpServletResponse;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/**
 * 全局异常处理。
 *
 * <p>注意：SSE 流式接口（{@code text/event-stream}）的异常不能在这里转成 JSON 响应体——
 * 响应头已经定死为事件流，写 JSON 会抛 {@code HttpMessageNotWritableException}，
 * 再触发容器 /error 页转发，把一次普通业务失败放大成一串无关堆栈。
 * 这类请求（以及任何响应已提交的请求）直接放行，交给调用方用事件流里的 error 事件告知前端。</p>
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(BusinessException.class)
    public ResponseEntity<ApiResponse<Void>> business(BusinessException e) {
        return ResponseEntity.badRequest().body(ApiResponse.fail(e.getCode(), e.getMessage()));
    }

    @ExceptionHandler(AccessDeniedException.class)
    public ResponseEntity<ApiResponse<Void>> denied(AccessDeniedException e) {
        return ResponseEntity.status(403).body(ApiResponse.fail(403, "无权限执行该操作"));
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ApiResponse<Void>> valid(MethodArgumentNotValidException e) {
        String m = e.getBindingResult().getFieldErrors().stream().findFirst()
                .map(x -> x.getField() + ": " + x.getDefaultMessage()).orElse("参数校验失败");
        return ResponseEntity.badRequest().body(ApiResponse.fail(400, m));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ApiResponse<Void>> other(Exception e, HttpServletResponse resp) {
        // SSE / 已提交响应：不再写 JSON 体（写了也发不出去，只会引发二次异常与堆栈噪音）
        if (resp != null) {
            String ct = String.valueOf(resp.getContentType());
            if (resp.isCommitted() || ct.contains("text/event-stream")) {
                return null;
            }
        }
        e.printStackTrace();
        return ResponseEntity.internalServerError()
                .body(ApiResponse.fail(500, e.getMessage() == null ? "服务器异常" : e.getMessage()));
    }
}
