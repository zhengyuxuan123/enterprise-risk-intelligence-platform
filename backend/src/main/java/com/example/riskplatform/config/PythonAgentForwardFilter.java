package com.example.riskplatform.config;

import com.example.riskplatform.security.CurrentUserService;
import com.example.riskplatform.service.DocumentAccessGuard;
import com.example.riskplatform.service.agent.TraceContext;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Reverse proxy for AI endpoints implemented by the Python agent.
 *
 * <p>The filter is deliberately narrow: only configured AI paths are forwarded,
 * and only callers that already have the AI permission can reach the Python
 * service. If the Python agent is unavailable, the filter now returns an
 * explicit 503 JSON response instead of falling through to Spring static
 * resource handling.</p>
 */
@Slf4j
@Component
public class PythonAgentForwardFilter extends OncePerRequestFilter {
    private static final String REQUIRED_AUTHORITY = "ai:analyze";

    private static final String HEADER_USER = "X-User-Id";
    private static final String HEADER_LEVEL = "X-Data-Level";
    private static final String HEADER_DEPTS = "X-Visible-Depts";
    private static final String HEADER_TRACE = "X-Trace-Id";

    private static final Set<String> FORWARDABLE_METHODS =
            Set.of("GET", "POST", "PUT", "PATCH", "DELETE");

    private final boolean enabled;
    private final String baseUrl;
    private final Set<String> forwardPaths;
    private final int timeoutSeconds;

    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(3))
            .version(HttpClient.Version.HTTP_1_1)
            .build();

    private final HttpClient streamHttp = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5))
            .version(HttpClient.Version.HTTP_1_1)
            .build();

    private final CurrentUserService current;
    private final DocumentAccessGuard acl;

    public PythonAgentForwardFilter(
            @Value("${app.agent-python.enabled:false}") boolean enabled,
            @Value("${app.agent-python.base-url:http://127.0.0.1:8081}") String baseUrl,
            @Value("${app.agent-python.forward-paths:/api/ai/agents,/api/ai/health,/api/ai/migration}")
            String forwardPaths,
            @Value("${app.agent-python.timeout-seconds:660}") int timeoutSeconds,
            CurrentUserService current,
            DocumentAccessGuard acl) {
        this.enabled = enabled;
        this.baseUrl = baseUrl == null ? "" : baseUrl.trim().replaceAll("/+$", "");
        this.forwardPaths = parsePaths(forwardPaths);
        this.timeoutSeconds = timeoutSeconds > 0 ? timeoutSeconds : 660;
        this.current = current;
        this.acl = acl;
        log.info("[PyAgent] forwardEnabled={} target={} paths={}",
                this.enabled, this.baseUrl.isEmpty() ? "-" : this.baseUrl, this.forwardPaths);
    }

    static boolean isForwardableMethod(String method) {
        return method != null && FORWARDABLE_METHODS.contains(method.toUpperCase(Locale.ROOT));
    }

    private static Set<String> parsePaths(String raw) {
        Set<String> out = new LinkedHashSet<>();
        if (raw == null || raw.isBlank()) return out;
        for (String s : raw.split("[,;\\s]+")) {
            String t = s.trim().replace("\"", "").replace("'", "");
            if (!t.isEmpty()) out.add(t);
        }
        return out;
    }

    private boolean pathAllowed(String uri) {
        if (uri == null) return false;
        for (String p : forwardPaths) {
            if (p.endsWith("/*")) {
                if (uri.startsWith(p.substring(0, p.length() - 1))) return true;
            } else if (p.equals(uri)) {
                return true;
            }
        }
        return false;
    }

    private static boolean authorized() {
        Authentication a = SecurityContextHolder.getContext().getAuthentication();
        if (a == null || !a.isAuthenticated()) return false;
        for (GrantedAuthority g : a.getAuthorities()) {
            String v = g.getAuthority();
            if (REQUIRED_AUTHORITY.equals(v) || "ROLE_ADMIN".equals(v)) return true;
        }
        return false;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
                                    FilterChain chain) throws ServletException, IOException {
        String uri = request.getRequestURI();
        if (!enabled || !pathAllowed(uri) || !isForwardableMethod(request.getMethod())) {
            chain.doFilter(request, response);
            return;
        }

        if (!authorized()) {
            log.debug("[PyAgent] {} lacks {}; leaving request to Spring Security", uri, REQUIRED_AUTHORITY);
            chain.doFilter(request, response);
            return;
        }

        byte[] body = request.getInputStream().readAllBytes();

        if (wantsEventStream(request) || uri.endsWith("/analyze/stream")) {
            streamForward(request, body, response, uri);
            return;
        }

        HttpResponse<byte[]> resp;
        try {
            resp = forward(request, body);
        } catch (Exception e) {
            String reason = describe(e);
            log.warn("[PyAgent] returning 503 because Python agent is unavailable: {}", reason);
            writeAgentUnavailable(response, uri, reason);
            return;
        }

        writeBufferedResponse(response, resp.statusCode(),
                resp.headers().firstValue("Content-Type").orElse("application/json"),
                resp.body());
        log.info("[PyAgent] {} -> Python {} ({} bytes)", uri, resp.statusCode(), resp.body().length);
    }

    private static boolean wantsEventStream(HttpServletRequest request) {
        String accept = request.getHeader("Accept");
        return accept != null && accept.toLowerCase(Locale.ROOT).contains("text/event-stream");
    }

    private void streamForward(HttpServletRequest request, byte[] body, HttpServletResponse response,
                               String uri) throws IOException {
        HttpResponse<InputStream> resp;
        try {
            resp = streamHttp.send(buildRequest(request, body, null),
                    HttpResponse.BodyHandlers.ofInputStream());
        } catch (Exception e) {
            String reason = describe(e);
            log.warn("[PyAgent] returning 503 because Python agent stream is unavailable: {}", reason);
            writeAgentUnavailable(response, uri, reason);
            return;
        }

        response.setStatus(resp.statusCode());
        response.setContentType(resp.headers().firstValue("Content-Type").orElse("text/event-stream"));
        response.setCharacterEncoding(StandardCharsets.UTF_8.name());

        long total = 0;
        try (InputStream in = resp.body()) {
            OutputStream out = response.getOutputStream();
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0) {
                out.write(buf, 0, n);
                out.flush();
                total += n;
            }
        } catch (Exception e) {
            log.debug("[PyAgent] {} stream interrupted: {}", uri, e.getClass().getSimpleName());
        }
        log.info("[PyAgent] {} -> Python stream complete ({} bytes)", uri, total);
    }

    private Long tryUserId() {
        try {
            return current.userId();
        } catch (Exception e) {
            return null;
        }
    }

    private Integer tryLevel() {
        try {
            return current.securityLevel();
        } catch (Exception e) {
            return null;
        }
    }

    private Set<Long> tryVisibleDepts() {
        if (acl == null) return null;
        try {
            return acl.currentVisibleDepts();
        } catch (Exception e) {
            return null;
        }
    }

    private String describe(Exception e) {
        String msg = e.getMessage();
        String kind = e.getClass().getSimpleName();
        if (msg == null || msg.isBlank()) {
            return kind + " (target " + baseUrl
                    + " unavailable: service not started, wrong port, or timeout)";
        }
        return kind + ": " + msg;
    }

    private void writeAgentUnavailable(HttpServletResponse response, String uri, String reason)
            throws IOException {
        String body = "{\"code\":503,\"message\":\"Python agent unavailable\","
                + "\"data\":{\"path\":\"" + json(uri) + "\",\"target\":\"" + json(baseUrl)
                + "\",\"reason\":\"" + json(reason) + "\"}}";
        writeBufferedResponse(response, HttpServletResponse.SC_SERVICE_UNAVAILABLE,
                "application/json", body.getBytes(StandardCharsets.UTF_8));
    }

    private static void writeBufferedResponse(HttpServletResponse response, int status,
                                              String contentType, byte[] bytes) throws IOException {
        if (response.isCommitted()) return;
        response.reset();
        response.setStatus(status);
        response.setContentType(contentType == null ? "application/json" : contentType);
        response.setCharacterEncoding(StandardCharsets.UTF_8.name());
        response.setContentLength(bytes.length);
        response.getOutputStream().write(bytes);
        response.getOutputStream().flush();
    }

    private static String json(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\r", "\\r").replace("\n", "\\n");
    }

    private HttpResponse<byte[]> forward(HttpServletRequest request, byte[] body) throws Exception {
        return http.send(buildRequest(request, body, Duration.ofSeconds(timeoutSeconds)),
                HttpResponse.BodyHandlers.ofByteArray());
    }

    HttpRequest buildRequest(HttpServletRequest request, byte[] body, Duration timeout) {
        String query = request.getQueryString();
        String target = baseUrl + request.getRequestURI()
                + (query == null || query.isEmpty() ? "" : "?" + query);

        HttpRequest.Builder b = HttpRequest.newBuilder()
                .uri(URI.create(target))
                .header("Content-Type",
                        request.getContentType() == null ? "application/json" : request.getContentType())
                .header(HEADER_TRACE, TraceContext.current());

        if (timeout != null) {
            b.timeout(timeout);
        }

        Long uid = tryUserId();
        if (uid != null) b.header(HEADER_USER, String.valueOf(uid));
        Integer level = tryLevel();
        b.header(HEADER_LEVEL, String.valueOf(level == null ? 1 : level));
        Set<Long> depts = tryVisibleDepts();
        if (depts != null) {
            b.header(HEADER_DEPTS, depts.stream().map(String::valueOf).collect(Collectors.joining(",")));
        }

        HttpRequest.BodyPublisher pub = HttpRequest.BodyPublishers.ofByteArray(body == null ? new byte[0] : body);
        String method = request.getMethod() == null ? "GET" : request.getMethod().toUpperCase(Locale.ROOT);
        switch (method) {
            case "POST" -> b.POST(pub);
            case "PUT" -> b.PUT(pub);
            case "DELETE" -> b.method("DELETE", pub);
            case "PATCH" -> b.method("PATCH", pub);
            default -> b.GET();
        }
        return b.build();
    }

}
