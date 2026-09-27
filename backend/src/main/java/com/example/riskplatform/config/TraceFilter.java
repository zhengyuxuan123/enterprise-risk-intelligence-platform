package com.example.riskplatform.config;

import com.example.riskplatform.service.agent.TraceContext;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

/**
 * 为每个请求绑定追溯号，并在响应头回传。
 *
 * <p>网关或前端可以在请求头 {@code X-Trace-Id} 里带上自己的流水号（比如工单号），
 * 整条链路会沿用；没带就现开一个。响应头回传同一个号，用户报障时只需报这 12 个字符。</p>
 */
@Component
@Order(Ordered.HIGHEST_PRECEDENCE + 10)
public class TraceFilter extends OncePerRequestFilter {

    @Override
    protected void doFilterInternal(HttpServletRequest req, HttpServletResponse resp, FilterChain chain)
            throws ServletException, IOException {
        String uri = req.getRequestURI();
        if (uri == null || !uri.startsWith("/api/")) {
            chain.doFilter(req, resp);
            return;
        }
        String incoming = req.getHeader(TraceContext.HEADER);
        String id = (incoming == null || incoming.isBlank()) ? TraceContext.newId() : incoming.trim();
        TraceContext.set(id);
        resp.setHeader(TraceContext.HEADER, id);
        try {
            chain.doFilter(req, resp);
        } finally {
            // 线程会被容器复用，不清会导致串号
            TraceContext.clear();
        }
    }
}
