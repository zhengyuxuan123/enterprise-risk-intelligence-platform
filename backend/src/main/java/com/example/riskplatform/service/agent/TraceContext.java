package com.example.riskplatform.service.agent;

import org.slf4j.MDC;

import java.security.SecureRandom;

/**
 * 全链路追踪号。
 *
 * <p>企业场景里，一份 AI 分析报告被质疑时，第一个问题一定是"把当时那次调用的完整过程调出来"。
 * 没有 traceId，日志、落库记录、工具调用之间无法互相定位，复盘只能靠时间猜。
 * 这里用一个短 ID 贯穿：日志（MDC）→ ai_analysis_trace 表 → 响应头 X-Trace-Id → 前端展示。</p>
 *
 * <p><b>为什么不直接用 UUID</b>：40 字符太长，用户抄下来报障不方便；这里用 12 位大小写字母数字，
 * 冲突概率在单企业量级下可忽略，必要时可由调用方通过 {@code X-Trace-Id} 请求头传入自己的流水号。</p>
 */
public final class TraceContext {

    public static final String HEADER = "X-Trace-Id";
    private static final String MDC_KEY = "traceId";
    private static final char[] CHARS = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789".toCharArray();
    private static final SecureRandom RND = new SecureRandom();

    private static final ThreadLocal<String> TL = new ThreadLocal<>();

    private TraceContext() {
    }

    public static String newId() {
        StringBuilder b = new StringBuilder(12);
        for (int i = 0; i < 12; i++) b.append(CHARS[RND.nextInt(CHARS.length)]);
        return b.toString();
    }

    /** 取当前线程的追踪号，没有就现开一个并绑定（幂等，不会重复生成）。 */
    public static String current() {
        String v = TL.get();
        if (v == null || v.isBlank()) {
            v = newId();
            set(v);
        }
        return v;
    }

    public static void set(String id) {
        if (id == null || id.isBlank()) return;
        TL.set(id);
        try {
            MDC.put(MDC_KEY, id);
        } catch (Exception ignored) {
        }
    }

    /**
     * 跨线程传递：把当前线程的追踪号带进工作线程，执行完清理。
     *
     * <p><b>为什么必须做</b>：SSE 流式分析、异步任务、评估任务都提交到线程池执行，
     * 而 {@link ThreadLocal} 不跨线程。不传递的话工作线程会自己新开一个追踪号，
     * 于是"响应头里的追溯号"和"落库留痕里的追溯号"对不上 —— 用户拿着追溯号根本查不到那次分析的留痕，
     * 全链路可追溯就是假的。</p>
     */
    public static Runnable propagate(Runnable task) {
        String id = current();
        return () -> {
            set(id);
            try {
                task.run();
            } finally {
                clear();
            }
        };
    }

    /** 请求结束必须调用，否则线程池复用线程会串号。 */
    public static void clear() {
        TL.remove();
        try {
            MDC.remove(MDC_KEY);
        } catch (Exception ignored) {
        }
    }
}
