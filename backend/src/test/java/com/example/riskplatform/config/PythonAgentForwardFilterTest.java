package com.example.riskplatform.config;

import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockFilterChain;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
import org.springframework.security.authentication.TestingAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;

import java.net.http.HttpRequest;
import java.time.Duration;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PythonAgentForwardFilterTest {
    private static PythonAgentForwardFilter filter() {
        return new PythonAgentForwardFilter(true, "http://127.0.0.1:8081", "/api/ai/*", 30, null, null);
    }

    private static PythonAgentForwardFilter unavailableFilter() {
        return new PythonAgentForwardFilter(true, "http://127.0.0.1:1", "/api/ai/*", 1, null, null);
    }

    @Test
    void deleteIsForwardable() {
        assertTrue(PythonAgentForwardFilter.isForwardableMethod("DELETE"),
                "AI history deletion must be proxied instead of falling through to Spring static resources");
    }

    @Test
    void putAndPatchAreForwardable() {
        assertTrue(PythonAgentForwardFilter.isForwardableMethod("PUT"));
        assertTrue(PythonAgentForwardFilter.isForwardableMethod("PATCH"));
    }

    @Test
    void getAndPostStillWork() {
        assertTrue(PythonAgentForwardFilter.isForwardableMethod("GET"));
        assertTrue(PythonAgentForwardFilter.isForwardableMethod("POST"));
    }

    @Test
    void caseIsIrrelevant() {
        assertTrue(PythonAgentForwardFilter.isForwardableMethod("delete"));
        assertTrue(PythonAgentForwardFilter.isForwardableMethod("Delete"));
    }

    @Test
    void unknownOrMissingMethodIsNotForwardable() {
        assertFalse(PythonAgentForwardFilter.isForwardableMethod("OPTIONS"));
        assertFalse(PythonAgentForwardFilter.isForwardableMethod("HEAD"));
        assertFalse(PythonAgentForwardFilter.isForwardableMethod(null));
    }

    @Test
    void deleteIsSentAsDeleteNotGet() {
        MockHttpServletRequest req = new MockHttpServletRequest("DELETE", "/api/ai/history/123");
        HttpRequest out = filter().buildRequest(req, new byte[0], Duration.ofSeconds(5));
        assertEquals("DELETE", out.method());
        assertEquals("http://127.0.0.1:8081/api/ai/history/123", out.uri().toString());
    }

    @Test
    void putAndPatchKeepTheirMethodAndBody() {
        byte[] body = "{\"a\":1}".getBytes();
        MockHttpServletRequest put = new MockHttpServletRequest("PUT", "/api/ai/x");
        put.setContent(body);
        HttpRequest out = filter().buildRequest(put, body, Duration.ofSeconds(5));
        assertEquals("PUT", out.method());

        MockHttpServletRequest patch = new MockHttpServletRequest("PATCH", "/api/ai/x");
        HttpRequest out2 = filter().buildRequest(patch, body, Duration.ofSeconds(5));
        assertEquals("PATCH", out2.method());
    }

    @Test
    void queryStringIsPreserved() {
        MockHttpServletRequest req = new MockHttpServletRequest("DELETE", "/api/ai/history");
        req.setQueryString("companyId=7");
        HttpRequest out = filter().buildRequest(req, new byte[0], Duration.ofSeconds(5));
        assertEquals("http://127.0.0.1:8081/api/ai/history?companyId=7", out.uri().toString());
    }

    @Test
    void timeoutIsAppliedWhenGivenAndOmittedWhenNull() {
        MockHttpServletRequest req = new MockHttpServletRequest("GET", "/api/ai/health");
        filter().buildRequest(req, new byte[0], Duration.ofSeconds(5));
        filter().buildRequest(req, new byte[0], null);
    }

    @Test
    void pathCanBeBuiltForAiPrefix() {
        MockHttpServletRequest in = new MockHttpServletRequest("GET", "/api/ai/rag/status");
        PythonAgentForwardFilter f = filter();
        assertEquals("http://127.0.0.1:8081/api/ai/rag/status",
                f.buildRequest(in, new byte[0], Duration.ofSeconds(5)).uri().toString());
    }

    @Test
    void unavailablePythonAgentReturns503InsteadOfFallingThroughToStaticResource() throws Exception {
        SecurityContextHolder.getContext().setAuthentication(new TestingAuthenticationToken(
                "admin", "n/a", List.of(new SimpleGrantedAuthority("ai:analyze"))));
        try {
            MockHttpServletRequest req = new MockHttpServletRequest("GET", "/api/ai/health");
            MockHttpServletResponse resp = new MockHttpServletResponse();
            unavailableFilter().doFilterInternal(req, resp, new MockFilterChain());

            assertEquals(503, resp.getStatus());
            assertTrue(resp.getContentAsString().contains("Python agent unavailable"));
        } finally {
            SecurityContextHolder.clearContext();
        }
    }
}
