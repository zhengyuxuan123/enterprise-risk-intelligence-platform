package com.example.riskplatform.config;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

/**
 * SPA 路由回退 —— 让 fat jar 能像一个真正的网站那样被使用。
 *
 * <p>背景：前端用的是 {@code createWebHistory()}（不是 hash 模式）。
 * jar 内部由 Spring Boot 从 {@code classpath:/static} 提供 {@code index.html}，
 * 于是「在页面里点菜单」没问题（Vue Router 接管），但**直接刷新或手输 URL**
 * （例如 {@code http://localhost:8080/dashboard}）会打到后端——后端根本没有这个接口，
 * 结果就是 404。本类把这些前端路由 forward 回 index.html，交给 Vue Router 解析。
 *
 * <p><b>为什么是显式列举而不是通配 {@code /{path}}：</b>
 * 通配会把 {@code /actuator}、{@code /error} 这类**非 /api 前缀的管理端点**
 * 也吞掉，排查时极难发现（请求会得到一个 HTML 而不是 JSON）。列清单虽然要人工同步，
 * 但行为完全可预测。
 *
 * <p><b>新增前端页面时记得同步这里</b>：路由清单见 {@code frontend/src/router.js} 的 routes。
 */
@Controller
public class SpaFallbackController {

    // 必须与 SecurityConfig 里 permitAll 的那批路径保持一致，
    // 否则会出现「先被 Security 拦 401，轮不到本类转发」的情况。
    // 注意：注解值必须是编译期常量数组，不能直接引用一个 String[] 常量变量。
    @GetMapping({
        "/login",
        "/dashboard",
        "/companies",
        "/metrics",
        "/import",
        "/rules",
        "/risks",
        "/complaints",
        "/competitors",
        "/knowledge",
        "/ai",
        "/audit",
        "/users",
    })
    public String forwardToIndex() {
        return "forward:/index.html";
    }
}
