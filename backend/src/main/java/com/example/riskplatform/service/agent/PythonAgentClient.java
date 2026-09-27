package com.example.riskplatform.service.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Java 侧调用 Python agent 的极简客户端。
 *
 * <p>Agent 栈（编排 / 检索 / 联网 / 模型调用）已整体迁到 Python 服务（默认 8081）。
 * 但 Java 侧的 CRUD 仍然需要在几个时间点上"通知"Python 一声：
 * 业务数据写库后要增量入索引、MQ 指标事件要唤起主动预警、知识库检索要取回结果。
 * 这个客户端就是那几条通知通道的唯一出口。</p>
 *
 * <p><b>为什么一律异步且静默失败</b>：这些都是"加分项"。
 * Python 没起、超时、返回错，都不该影响业务写库或规则引擎落库 ——
 * 否则一次 Python 抖动会把「录入一条指标」变成 500，这是本末倒置。
 * 失败只记 debug 日志，索引一致性由 Python 侧的定时 reconcile 兜底。</p>
 */
@Slf4j
@Service
public class PythonAgentClient {

    private final String baseUrl;
    private final boolean enabled;
    private final ObjectMapper mapper = new ObjectMapper();
    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(3))
            // Uvicorn 默认只接 HTTP/1.1；禁止 JDK HttpClient 的 h2c 升级探测。
            .version(HttpClient.Version.HTTP_1_1)
            .build();
    private final ExecutorService pool = Executors.newFixedThreadPool(2, r -> {
        Thread t = new Thread(r, "py-agent-notify");
        t.setDaemon(true);
        return t;
    });

    public PythonAgentClient(@Value("${app.agent-python.base-url:http://127.0.0.1:8081}") String baseUrl,
                             @Value("${app.agent-python.enabled:false}") boolean enabled) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.enabled = enabled;
    }

    /** 是否可用：开关没开就不发任何请求，省掉一次注定失败的网络往返。 */
    public boolean available() {
        return enabled && !baseUrl.isBlank();
    }

    /**
     * 异步 POST，不关心返回值，失败静默。
     *
     * <p>用独立线程池而不是 {@code CompletableFuture.runAsync} 的默认池：
     * 后者是公共 ForkJoinPool，被拖慢会连带影响 Spring 的异步任务。</p>
     */
    public void postAsync(String pathAndQuery) {
        if (!available()) return;
        pool.submit(() -> {
            try {
                HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + pathAndQuery))
                        .timeout(Duration.ofSeconds(10))
                        .header("Accept", "application/json")
                        .POST(HttpRequest.BodyPublishers.noBody())
                        .build();
                http.send(req, HttpResponse.BodyHandlers.discarding());
            } catch (Exception e) {
                log.debug("[py-agent] 通知失败 {} : {}", pathAndQuery, e.getMessage());
            }
        });
    }

    /** 同步 GET，取 {@code data} 节点；失败返回 null。仅在明确需要结果时用（如知识库检索）。 */
    public JsonNode get(String pathAndQuery) {
        if (!available()) return null;
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + pathAndQuery))
                    .timeout(Duration.ofSeconds(15))
                    .header("Accept", "application/json")
                    .GET()
                    .build();
            HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() != 200) return null;
            JsonNode root = mapper.readTree(res.body());
            return root.path("data").isObject() ? root.get("data") : null;
        } catch (Exception e) {
            log.debug("[py-agent] 查询失败 {} : {}", pathAndQuery, e.getMessage());
            return null;
        }
    }

    /** 同步 JSON POST。可靠任务由调用方记录失败并重试，因此这里不能静默吞错。 */
    public JsonNode post(String pathAndQuery, String jsonBody) {
        if (!available()) throw new IllegalStateException("Python Agent 未启用");
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + pathAndQuery))
                    .timeout(Duration.ofSeconds(60))
                    .header("Accept", "application/json")
                    .header("Content-Type", "application/json; charset=UTF-8")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody == null ? "{}" : jsonBody))
                    .build();
            HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
            JsonNode root = mapper.readTree(res.body());
            if (res.statusCode() < 200 || res.statusCode() >= 300 || root.path("code").asInt(-1) != 0) {
                throw new IllegalStateException(root.path("message").asText("HTTP " + res.statusCode()));
            }
            return root.get("data");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("调用 Python Agent 被中断", e);
        } catch (Exception e) {
            throw new IllegalStateException("调用 Python Agent 失败: " + e.getMessage(), e);
        }
    }
}
